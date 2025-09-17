from omegaconf import OmegaConf
import os
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.workers import Worker
from RL2.datasets import get_tensor_dict, pack_tensor_dicts
from RL2.utils.comm import split_and_scatter_list, gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log


class Rollout(Worker):

    def __init__(self, config):
        super().__init__(config, None)
        
        self.prepare_environment_variables()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.prepare_environment()

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = Engine(
                model_path=config.model_name,
                dtype=config.dtype,
                tp_size=self.device_mesh["tp"].size(),
                mem_fraction_static=config.gpu_memory_utilization,
                enable_memory_saver=True,
                port=config.base_port + dist.get_rank()
            )
        
            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % self.config.tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {self.config.tp_size}."
        self.dp_size = world_size // self.config.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.tp_size)
        )

    def prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def initialize_state_dict(self, state_text):

        states = self.tokenizer.encode(state_text, add_special_tokens=False)
        return {
            "states": states,
            "actions": len(states) * [0],
            "action_mask": len(states) * [0],
            "logps": len(states) * [0],
            "rewards": len(states) * [0]
        }
    
    def get_tensor_dict(self, state_dict):

        tensor_dict = get_tensor_dict(
            state_dict["states"],
            state_dict["actions"],
            state_dict["action_mask"]
        )
        tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
        tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
        return tensor_dict
        
    async def rollout(self, ex, train):

        state_text = (
            ex["prompt"] if "prompt" in ex else
            await self.env.reset(ex["extra_info"])
        )
        state_dict = self.initialize_state_dict(state_text)
        env_response = {"extra_info": ex["extra_info"]}
        tensor_dicts = []
        metric = defaultdict(list)
        scores = []
        for turn in range(1, self.config.max_turns + 1):

            llm_response = await self.llm.async_generate(
                input_ids=state_dict["states"],
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params,
                return_logprob=True
            )

            action_text = llm_response["text"]
            env_response = await self.env.step(
                state_text, action_text, env_response["extra_info"]
            )

            meta_info = llm_response["meta_info"]
            logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
            state_dict["states"].extend(action)
            state_dict["actions"].extend(action)
            state_dict["action_mask"].extend(len(action) * [1])
            state_dict["logps"].extend(logp)
            state_dict["rewards"].extend((len(action) - 1) * [0] + [env_response["reward"]])
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )
            scores.append(env_response["score"])

            if turn == self.config.max_turns or env_response["done"]:
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                break
            if env_response["next_state"].startswith(state_text + action_text):
                state_dict_delta = self.initialize_state_dict(
                    env_response["next_state"][len(state_text + action_text):]
                )
                for k, v in state_dict_delta.items():
                    state_dict[k].extend(v)
            else:
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                state_dict = self.initialize_state_dict(env_response["next_state"])
            state_text = env_response["next_state"]

        metric["n_turns"].append(turn)
        metric["scores"].append(sum(scores))

        return tensor_dicts, metric

    async def vectorized_rollout(self, num_episodes, train):
        """
        Vectorized rollout for environments that support batch operations.
        Uses batched generation with SGLang for improved efficiency.

        Args:
            num_episodes: Target number of episodes to collect
            train: Whether this is training or evaluation

        Returns:
            Tuple of (all_tensor_dicts, metrics)
        """
        import importlib.util

        # Load vectorized environment
        spec = importlib.util.spec_from_file_location("vec_env", self.config.env_path)
        env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_module)

        # Get batch size from vectorized environment
        batch_size = getattr(env_module, 'NUM_ENVS', 16)
        num_batches = (num_episodes + batch_size - 1) // batch_size

        all_tensor_dicts = []
        all_metrics = defaultdict(list)

        for batch_idx in range(num_batches):
            # Reset environments to get initial observations
            batch_observations = await env_module.reset(None)
            if isinstance(batch_observations, str):
                batch_observations = [batch_observations] * batch_size

            batch_state_dicts = []
            batch_env_responses = []
            batch_tensor_dicts = []
            batch_metrics = defaultdict(list)
            batch_scores = []
            episode_completed = [False] * batch_size  # Track which episodes have completed
            batch_state_texts = []  # Track state text for each environment

            # Initialize state dicts for each environment in the batch
            for obs in batch_observations:
                state_dict = self.initialize_state_dict(obs)
                batch_state_dicts.append(state_dict)
                batch_env_responses.append({"extra_info": None})
                batch_scores.append([])
                batch_state_texts.append(obs)

            for turn in range(1, self.config.max_turns + 1):
                # Prepare batch input_ids for batched generation (all envs)
                batch_input_ids = [batch_state_dicts[i]["states"] for i in range(batch_size)]

                # Batched generation using SGLang
                llm_responses = await self.llm.async_generate(
                    input_ids=batch_input_ids,
                    sampling_params=self.train_sampling_params if train else self.test_sampling_params,
                    return_logprob=True
                )

                # Handle case where response is not a list (single response)
                if not isinstance(llm_responses, list):
                    llm_responses = [llm_responses]

                # Extract actions for batch environment step (all envs)
                batch_actions = [response["text"] for response in llm_responses]
                batch_extra_info = [batch_env_responses[i]["extra_info"] for i in range(batch_size)]

                # Batch environment step for all environments
                env_responses = await env_module.step(
                    batch_observations, batch_actions, batch_extra_info
                )

                # Process responses for each environment
                for i, llm_response in enumerate(llm_responses):
                    state_dict = batch_state_dicts[i]

                    env_response = {
                        "next_state": env_responses["next_state"][i],
                        "reward": env_responses["reward"][i],
                        "done": env_responses["done"][i],
                        "extra_info": env_responses["extra_info"][i]
                    }

                    meta_info = llm_response["meta_info"]
                    logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))

                    # Update state dict
                    state_dict["states"].extend(action)
                    state_dict["actions"].extend(action)
                    state_dict["action_mask"].extend(len(action) * [1])
                    state_dict["logps"].extend(logp)
                    state_dict["rewards"].extend((len(action) - 1) * [0] + [env_response["reward"]])

                    # Collect metrics
                    batch_metrics["response_length"].append(meta_info["completion_tokens"])
                    batch_metrics["length_clip_ratio"].append(
                        meta_info["finish_reason"]["type"] == "length"
                    )
                    batch_scores[i].append(env_response["reward"])

                    action_text = batch_actions[i]
                    state_text = batch_state_texts[i]

                    # Check if episode is done
                    if turn == self.config.max_turns or env_response["done"]:
                        # Only append to batch_tensor_dicts if this episode hasn't completed before
                        if not episode_completed[i]:
                            batch_tensor_dicts.append(self.get_tensor_dict(state_dict))
                            episode_completed[i] = True

                        # Re-initialize state dict for next episode
                        batch_state_dicts[i] = self.initialize_state_dict(env_response["next_state"])
                        batch_scores[i] = []  # Reset scores for new episode
                        batch_state_texts[i] = env_response["next_state"]
                    else:
                        # Handle state transitions with delta updates
                        if env_response["next_state"].startswith(state_text + action_text):
                            state_dict_delta = self.initialize_state_dict(
                                env_response["next_state"][len(state_text + action_text):]
                            )
                            for k, v in state_dict_delta.items():
                                state_dict[k].extend(v)
                        else:
                            # Complete current trajectory and start new one
                            if not episode_completed[i]:
                                batch_tensor_dicts.append(self.get_tensor_dict(state_dict))
                                episode_completed[i] = True
                            batch_state_dicts[i] = self.initialize_state_dict(env_response["next_state"])

                        batch_state_texts[i] = env_response["next_state"]
                        # Update observation for next turn
                        batch_observations[i] = env_response["next_state"]

            # Collect episode metrics
            for i, scores in enumerate(batch_scores):
                batch_metrics["n_turns"].append(turn)
                batch_metrics["scores"].append(sum(scores))

            # Add batch results to overall collection
            all_tensor_dicts.extend(batch_tensor_dicts)
            for key, values in batch_metrics.items():
                all_metrics[key].extend(values)

        return all_tensor_dicts, dict(all_metrics)

    def is_vectorized_env(self):
        """Check if the environment is vectorized by examining env_path."""
        if not hasattr(self.config, 'env_path') or self.config.env_path is None:
            return False

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("env_check", self.config.env_path)
            env_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(env_module)

            # Check for vectorized environment markers
            return (hasattr(env_module, 'NUM_ENVS') and
                    hasattr(env_module, 'vec_env') and
                    hasattr(env_module.vec_env, 'num_envs'))
        except:
            return False

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # The data is distributed from rank 0 before each worker operation
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            # Check if we're using vectorized environments
            if self.is_vectorized_env():
                # For vectorized environments, ignore data_list and collect episodes directly
                total_target_episodes = getattr(self.config, 'prompts_per_rollout', len(data_list))
                # Divide episodes across devices
                target_episodes = total_target_episodes // self.device_mesh["dp"].size()

                loop = asyncio.get_event_loop()
                all_tensor_dicts, metrics = loop.run_until_complete(
                    self.vectorized_rollout(target_episodes, train)
                )
                outputs = [(all_tensor_dicts, metrics)]
            else:
                # Traditional individual rollout processing
                data_list = split_and_scatter_list(
                    data_list, self.device_mesh["dp"]
                )
                loop = asyncio.get_event_loop()
                outputs = loop.run_until_complete(
                    tqdm.gather(
                        *(self.rollout(ex, train) for ex in data_list),
                        desc="Rollout",
                        position=1,
                        leave=False,
                        disable=(dist.get_rank() != 0)
                    )
                )

            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:

            if self.is_vectorized_env():
                # Handle vectorized environment outputs
                all_tensor_dicts, metrics = outputs[0]

                suffix = "train" if train else "test"
                formatted_metrics = {
                    f"{k}/{suffix}": v if isinstance(v, list) else [v]
                    for k, v in metrics.items()
                }
                gather_and_log(formatted_metrics, self.device_mesh["dp"], step)

                if not train:
                    return

                # For vectorized environments, all_tensor_dicts is already a flat list
                # Reshape into the expected format
                all_tensor_dicts_grouped = [[td] for td in all_tensor_dicts]

            else:
                # Traditional processing
                all_tensor_dicts_grouped, metrics = map(list, zip(*outputs))

                suffix = "train" if train else "test"
                metrics = {
                    f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                    for k in metrics[0].keys()
                }
                gather_and_log(metrics, self.device_mesh["dp"], step)

                if not train:
                    return

            all_tensor_dicts_grouped = gather_and_concat_list(
                all_tensor_dicts_grouped, self.device_mesh["dp"]
            )

            if dist.get_rank() == 0:

                group_size = self.config.responses_per_prompt
                if group_size > 1 and self.config.dynamic_filtering and not self.is_vectorized_env():

                    rewards = torch.FloatTensor([
                        sum([td["rewards"].sum().item() for td in tensor_dicts])
                        for tensor_dicts in all_tensor_dicts_grouped
                    ]).view(-1, group_size)
                    are_filtered = rewards.std(-1) == 0
                    all_tensor_dicts_grouped = sum([
                        all_tensor_dicts_grouped[idx * group_size:(idx + 1) * group_size]
                        for idx, is_filtered in enumerate(are_filtered)
                        if not is_filtered
                    ], [])
                    wandb.log({
                        "dynamic_filtering_ratio": are_filtered.float().mean().item()
                    }, step=step)

                tensor_dicts = sum(all_tensor_dicts_grouped, [])
                tensor_dict = pack_tensor_dicts(tensor_dicts)
                seqs = torch.LongTensor([
                    len(tensor_dicts) for tensor_dicts in all_tensor_dicts_grouped
                ])
                cu_seqs = torch.cumsum(
                    torch.cat((torch.LongTensor([0]), seqs)), dim=0
                )

                return tensor_dict, cu_seqs

        return None, None
        
    @time_logger("update_rollout")
    def update(self, actor, step):

        torch.cuda.empty_cache()
        dist.barrier()
        # or llm.resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()
        
        for idx, (name, tensor) in enumerate(actor.state_dict.items()):
            tensor = tensor.to(torch.cuda.current_device())
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=(idx == len(actor.state_dict) - 1)
                )
        actor.state_dict.clear()
        dist.barrier()