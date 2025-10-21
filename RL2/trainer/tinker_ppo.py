import asyncio
from collections import deque
from typing import List, Dict, Any
import logging

import hydra
import numpy as np
import torch
import wandb
from tqdm import tqdm

import tinker
from tinker.types import SamplingParams, ModelInput, Datum, AdamParams
from tinker.types.tensor_data import TensorData

# Reuse from RL2
from RL2.trainer import Trainer
from RL2.datasets import RLDataset, get_dataloader
from RL2.utils.algorithms import compute_advantages

logger = logging.getLogger(__name__)


class TinkerPPOTrainer(Trainer):
    """
    Tinker-based PPO/REINFORCE trainer for RL2.

    Features:
    - Async overlap between generation and training (~1.8x speedup)
    - Reuses RL2's dataset, config, and advantage computation
    - Uses Tinker API for distributed training (no torchrun needed)
    """

    def __init__(self, config):
        super().__init__(config)  # RL2's base: wandb, checkpointing, config handling

        self.train_dataloader = self.get_dataloader(True)
        self.test_dataloader = self.get_dataloader(False) if config.test_data.path else None

        # Tinker clients (initialized in setup_tinker)
        self.training_client = None
        self.service_client = None
        self.current_sampling_client = None

        # Tokenizer (will be set in setup_tinker)
        self.tokenizer = None

        # Environment module (for reward computation)
        self.env_module = None

    def get_dataloader(self, train: bool):
        """Reuse RL2's dataset loading"""
        dataset = RLDataset(
            self.config.train_data if train else self.config.test_data,
            tokenizer=None  # Will use Tinker's tokenizer
        )
        return get_dataloader(
            dataset,
            self.config.train_data.prompts_per_rollout if train else len(dataset)
        )

    async def setup_tinker(self):
        """Initialize Tinker clients"""
        logger.info("Setting up Tinker clients...")

        self.service_client = tinker.ServiceClient(
            base_url=self.config.tinker.get('base_url')
        )

        self.training_client = await self.service_client.create_lora_training_client_async(
            base_model=self.config.tinker.model_name,
            rank=self.config.tinker.lora_rank
        )

        # Get tokenizer from Tinker
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tinker.model_name,
            trust_remote_code=True
        )

        # Load environment module if configured (like RL2's env_path)
        if hasattr(self.config.rollout, 'env_module') and self.config.rollout.env_module:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "env_module", self.config.rollout.env_module
            )
            self.env_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.env_module)
            logger.info(f"Loaded environment module: {self.config.rollout.env_module}")

        # Create initial sampling client
        save_future = await self.training_client.save_weights_for_sampler_async('init')
        save_result = await save_future.result_async()
        self.current_sampling_client = self.service_client.create_sampling_client(save_result.path)

        logger.info("Tinker clients ready!")

    def tokenize_prompt(self, prompt: str) -> List[int]:
        """Tokenize a prompt string"""
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    async def compute_reward(self, data: Dict[str, Any], response_tokens: List[int]) -> float:
        """
        Compute reward for a response using environment step function.

        Similar to RL2's rollout, this calls the environment's step() function
        to get the reward (e.g., using math_verify for math problems).

        Requires rollout.env_module to be configured.
        """
        if not (hasattr(self.config.rollout, 'env_module') and self.config.rollout.env_module):
            raise ValueError(
                "rollout.env_module must be configured. "
                "Set rollout.env_module to path of environment file (e.g., RL2/envs/orz.py)"
            )

        response_text = self.tokenizer.decode(response_tokens)

        # Call environment step function (like RL2/envs/orz.py)
        result = await self.env_module.step(
            state=None,  # Not used in single-turn
            action=response_text,
            extra_info=data.get('extra_info', {})
        )
        return float(result['reward'])

    async def collect_episode_batch(self, sampling_client, data_list: List[Dict]) -> List[Dict]:
        """
        Collect episodes for a batch - runs async!
        All episodes sampled concurrently.
        """
        sampling_params = SamplingParams(
            max_tokens=self.config.tinker.max_tokens,
            temperature=self.config.rollout.train_sampling_params.temperature
        )

        async def sample_one(data):
            """Sample a single episode"""
            prompt = data['prompt']
            prompt_tokens = self.tokenize_prompt(prompt)

            # Sample from Tinker
            result = await sampling_client.sample_async(
                prompt=ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=1,
                sampling_params=sampling_params
            )

            response_tokens = result.sequences[0].tokens
            logprobs = result.sequences[0].logprobs

            # Get reward
            reward = await self.compute_reward(data, response_tokens)

            return {
                'prompt': prompt,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'logprobs': logprobs,
                'reward': reward,  # Base reward (will apply KL penalty later)
                'extra_info': data.get('extra_info', {}),
            }

        # Run all samples concurrently
        episodes = await asyncio.gather(*[sample_one(data) for data in data_list])
        return episodes

    def compute_advantages_from_rewards(self, rewards: List[float]) -> List[float]:
        """
        Compute advantages from rewards using RL2's advantage computation logic.

        Matches RL2/RL2/utils/algorithms.py:compute_reinforce_adv()

        Two modes:
        - global_norm=False (GRPO): Normalize within groups (per prompt)
          advantages = rewards - mean(rewards_per_prompt)
        - global_norm=True (ReBN): Normalize across entire batch
          advantages = rewards - mean(all_rewards)
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        responses_per_prompt = self.config.train_data.responses_per_prompt

        # Reshape to [num_prompts, responses_per_prompt]
        rewards_reshaped = rewards_tensor.view(-1, responses_per_prompt)

        if self.config.adv.global_norm:
            # Global normalization (ReBN): use mean/std across entire batch
            baseline = rewards_reshaped.mean()
            std = rewards_reshaped.std()
            logger.debug(f"Global norm: mean={baseline:.3f}, std={std:.3f}")
        else:
            # GRPO: normalize within each group (per prompt)
            # baseline shape: [num_prompts, 1]
            baseline = rewards_reshaped.mean(dim=-1, keepdim=True)
            std = rewards_reshaped.std(dim=-1, keepdim=True)
            logger.debug(f"GRPO: per-prompt normalization, avg_mean={baseline.mean():.3f}")

        # Compute advantages
        advantages = rewards_reshaped - baseline

        if self.config.adv.norm_var:
            # Divide by std for variance normalization
            advantages /= (std + torch.finfo(advantages.dtype).eps)

        # Flatten back to list
        return advantages.view(-1).tolist()

    async def train_on_episodes(self, episodes: List[Dict], ref_logprobs: List[torch.Tensor] = None) -> Dict[str, float]:
        """
        Train on collected episodes - runs async!
        Can overlap with sampling for next batch.

        Args:
            episodes: List of episode dictionaries
            ref_logprobs: Optional reference logprobs for KL penalty (from ref model)
        """
        # 1. Apply KL penalty to rewards if enabled
        rewards = [ep['reward'] for ep in episodes]

        if self.config.tinker.kl_penalty_coef > 0 and ref_logprobs is not None:
            # Compute KL penalty and subtract from rewards
            # KL = log_probs_old - log_probs_ref
            kl_penalties = []
            for ep, ref_logprob in zip(episodes, ref_logprobs):
                sample_logprob = torch.tensor(ep['logprobs'])
                kl = (sample_logprob - ref_logprob).sum().item()
                kl_penalties.append(kl)

            # Subtract KL penalty from rewards
            rewards = [r - self.config.tinker.kl_penalty_coef * kl for r, kl in zip(rewards, kl_penalties)]
            logger.debug(f"Applied KL penalty: mean_kl={np.mean(kl_penalties):.4f}")

        # 2. Compute advantages
        advantages = self.compute_advantages_from_rewards(rewards)

        # 2. Prepare Tinker training data
        training_data = []
        for episode, advantage in zip(episodes, advantages):
            prompt_len = len(episode['prompt_tokens'])
            tokens = episode['prompt_tokens'] + episode['response_tokens']

            # Advantages: 0 for prompt tokens, advantage for response tokens
            adv_vec = [0.0] * (prompt_len - 1) + [advantage] * len(episode['response_tokens'])

            # Logprobs: 0 for prompt tokens, actual logprobs for response
            logprob_vec = [0.0] * (prompt_len - 1) + episode['logprobs']

            datum = Datum(
                model_input=ModelInput.from_ints(tokens=tokens[:-1]),
                loss_fn_inputs={
                    'target_tokens': TensorData.from_torch(torch.tensor(tokens[1:])),
                    'logprobs': TensorData.from_torch(torch.tensor(logprob_vec)),
                    'advantages': TensorData.from_torch(torch.tensor(adv_vec)),
                }
            )
            training_data.append(datum)

        # 3. Forward-backward pass
        fwd_bwd_future = await self.training_client.forward_backward_async(
            training_data,
            loss_fn=self.config.tinker.loss_fn
        )
        fwd_bwd_result = await fwd_bwd_future.result_async()

        # 4. Optimizer step
        adam_params = AdamParams(
            learning_rate=self.config.tinker.learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8
        )
        optim_future = await self.training_client.optim_step_async(adam_params)
        await optim_future.result_async()

        # 5. Compute metrics (following tinker-cookbook's metrics.py)
        train_logprobs = torch.cat([
            output['logprobs'].to_torch()[-len(episodes[i]['response_tokens']):]
            for i, output in enumerate(fwd_bwd_result.loss_fn_outputs)
        ])
        sample_logprobs = torch.cat([torch.tensor(ep['logprobs']) for ep in episodes])

        # KL divergence metrics (tinker-cookbook style)
        logprob_diffs = sample_logprobs - train_logprobs
        kl_sample_train_v1 = logprob_diffs.mean().item()  # First order KL approximation
        kl_sample_train_v2 = 0.5 * (logprob_diffs ** 2).mean().item()  # Second order KL approximation

        # Entropy
        entropy_sample = -sample_logprobs.mean().item()

        # Base reward metrics (before KL penalty)
        base_rewards = [ep['reward'] for ep in episodes]

        metrics = {
            # Loss
            'train/loss': fwd_bwd_result.metrics.get('loss', 0.0),

            # Rewards (after KL penalty if applied)
            'train/reward_mean': np.mean(rewards),
            'train/reward_std': np.std(rewards),
            'train/reward_max': np.max(rewards),
            'train/reward_min': np.min(rewards),

            # Base rewards (before KL penalty)
            'train/base_reward_mean': np.mean(base_rewards),

            # KL divergence metrics (tinker-cookbook style)
            'optim/kl_sample_train_v1': kl_sample_train_v1,
            'optim/kl_sample_train_v2': kl_sample_train_v2,
            'optim/entropy': entropy_sample,

            # Episode statistics
            'train/episode_len_mean': np.mean([len(ep['response_tokens']) for ep in episodes]),
            'train/episode_len_std': np.std([len(ep['response_tokens']) for ep in episodes]),
            'train/num_episodes': len(episodes),
        }

        # Add KL penalty metrics if enabled
        if self.config.tinker.kl_penalty_coef > 0 and ref_logprobs is not None:
            kl_penalties = []
            for ep, ref_logprob in zip(episodes, ref_logprobs):
                sample_logprob = torch.tensor(ep['logprobs'])
                kl = (sample_logprob - ref_logprob).sum().item()
                kl_penalties.append(kl)
            metrics['train/kl_penalty_mean'] = np.mean(kl_penalties)

        return metrics

    async def async_train_loop(self):
        """
        Main async training loop with overlap between sampling and training.

        Pattern:
        1. Start sampling batch N
        2. While sampling batch N, train on batch N-1
        3. Repeat with overlap for ~1.8x speedup
        """
        step = 0

        # Create iterator
        data_iter = iter(self.train_dataloader)

        # Start sampling first batch
        try:
            first_batch = next(data_iter)
        except StopIteration:
            logger.warning("Empty training dataloader!")
            return

        sampling_task = asyncio.create_task(
            self.collect_episode_batch(self.current_sampling_client, first_batch)
        )

        for epoch in range(self.config.trainer.n_epochs):
            pbar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.config.trainer.n_epochs}",
                disable=False
            )

            while True:
                step += 1

                # Wait for current sampling to finish
                current_episodes = await sampling_task

                # Start next sampling immediately (overlap with training)
                try:
                    next_batch = next(data_iter)
                    sampling_task = asyncio.create_task(
                        self.collect_episode_batch(self.current_sampling_client, next_batch)
                    )
                    has_next = True
                except StopIteration:
                    has_next = False

                # Train on current episodes (overlaps with next batch sampling!)
                metrics = await self.train_on_episodes(current_episodes)
                metrics['step'] = step
                metrics['epoch'] = epoch

                # Log to wandb
                if self.config.trainer.use_wandb:
                    wandb.log(metrics)

                pbar.set_postfix({
                    'reward': f"{metrics['train/reward_mean']:.3f}",
                    'kl_v1': f"{metrics['optim/kl_sample_train_v1']:.4f}",
                    'entropy': f"{metrics['optim/entropy']:.2f}"
                })
                pbar.update(1)

                # Periodically update sampling client and save checkpoint
                if self.config.trainer.save_freq and step % self.config.trainer.save_freq == 0:
                    logger.info(f"Saving checkpoint at step {step}...")

                    # Save for sampling
                    save_future = await self.training_client.save_weights_for_sampler_async(f'step{step}')
                    save_result = await save_future.result_async()
                    self.current_sampling_client = self.service_client.create_sampling_client(save_result.path)

                    # Save training state
                    await self.training_client.save_state_async(f'step{step}')

                    logger.info(f"Checkpoint saved: {save_result.path}")

                # Test if configured
                if self.test_dataloader and self.config.trainer.test_freq:
                    if step % self.config.trainer.test_freq == 0:
                        await self.run_test(step)

                if not has_next:
                    break

            pbar.close()

    async def run_test(self, step: int):
        """Run evaluation on test set"""
        logger.info(f"Running test at step {step}...")

        test_rewards = []
        for data_list in self.test_dataloader:
            episodes = await self.collect_episode_batch(self.current_sampling_client, data_list)
            test_rewards.extend([ep['reward'] for ep in episodes])

        test_metrics = {
            'test/reward_mean': np.mean(test_rewards),
            'test/reward_std': np.std(test_rewards),
            'step': step,
        }

        if self.config.trainer.use_wandb:
            wandb.log(test_metrics)

        logger.info(f"Test results: reward_mean={test_metrics['test/reward_mean']:.3f}")

    async def train(self):
        """Main entry point"""
        await self.setup_tinker()

        if self.config.tinker.enable_async_overlap:
            logger.info("Starting async training with generation/training overlap...")
            await self.async_train_loop()
        else:
            logger.info("Starting sync training (no overlap)...")
            # Could implement sync version here if needed
            await self.async_train_loop()  # For now, just use async


@hydra.main(config_path="config", config_name="tinker_ppo", version_base=None)
async def main(config):
    """
    Entry point for Tinker-based RL2 training.

    Usage:
        python -m RL2.trainer.tinker_ppo \\
            train_data.path=prompts.jsonl \\
            tinker.model_name=Qwen/Qwen3-8B-Base
    """
    # No need for initialize_global_process_group - Tinker handles distributed training!

    trainer = TinkerPPOTrainer(config)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
