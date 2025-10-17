from collections import defaultdict
import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.algorithms import compute_approx_kl
from RL2.utils.logging import (
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class MegatronActor(MegatronWorker):
    
    def __init__(self, config, train: bool):
        super().__init__(config, train)
        # TODO: wrap_with_ddp=train?
        self.model = self.bridge.get_model(wrap_with_ddp=True)
        self.prepare_model_optimizer()

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)
        self.load_model_to_gpu()

        prefix = "old" if self.train else "ref"
        for model in self.model:
            model.eval()
        def f(minibatch, cu_seqlens, logits, non_loss_data=True):

            compute_logps_and_entropy(
                logits / getattr(self.config, "temperature", 1.0),
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                prefix
            )
            return gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
        
        minibatches = self.forward_backward(f, minibatches)

        if not self.train:
            self.offload_model_to_cpu()
        return self.gather_data(minibatches)

    @time_logger("update_actor")
    def sft_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)

        total_actions, total_sequences = count_total(
            minibatches,
            ("action_mask", "eos_mask"),
            mpu.get_data_parallel_group()
        )

        def f(minibatch, cu_seqlens, logits):

            compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group()
            )
            minibatch = gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            return self.scale_loss(loss), 1, {"loss": [loss.item()]}

        metrics, grad_norm = self.forward_backward(f, minibatches)
        metrics["grad_norm"] = [grad_norm]
        gather_and_log(metrics, step, mpu.get_data_parallel_group())

    @time_logger("update_actor")
    def dpo_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict, pair=True)

        total_pairs = count_total(
            minibatches, "eos_mask", mpu.get_data_parallel_group()
        ) // 2

        def f(minibatch, cu_seqlens, logits):

            compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group()
            )
            minibatch = gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
            chosen_rewards, rejected_rewards = self.config.beta * (
                minibatch["logps"] - minibatch["ref_logps"]
            ).sum(-1).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / total_pairs
            metric = {
                "rewards/chosen": chosen_rewards.tolist(),
                "rewards/rejected": rejected_rewards.tolist(),
                "rewards/margin": reward_margins.tolist(),
                "loss": [loss.item()],
                "accuracy": (reward_margins > 0).tolist()
            }
            return self.scale_loss(loss), 1, metric

        metrics, grad_norm = self.forward_backward(f, minibatches)
        metrics["grad_norm"] = [grad_norm]
        gather_and_log(metrics, step, mpu.get_data_parallel_group())

    @time_logger("update_actor")
    def ppo_update(self, tensor_dict, step):
        if step < self.config.freeze_steps:
            return
        batches = self.scatter_data(tensor_dict, pack_minibatches=True)
        self.load_model_to_gpu()

        for model in self.model:
            model.train()
        metrics = defaultdict(list)
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                mpu.get_data_parallel_group()
            )

            def f(minibatch, cu_seqlens, logits):
            
                compute_logps_and_entropy(
                    logits / getattr(self.config, "temperature", 1.0),
                    minibatch,
                    mpu.get_tensor_model_parallel_group(),
                    return_entropy=True
                )
                minibatch = gather_along_cp(
                    minibatch,
                    mpu.get_context_parallel_group(),
                    cu_seqlens
                )

                ratio = torch.exp(
                    minibatch["logps"] - minibatch.get(
                        "old_logps", minibatch["logps"].detach()
                    )
                )
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.config.clip, 1 + self.config.clip
                )
                objective = minibatch["advantages"] * ratio
                clipped_objective = minibatch["advantages"] * clipped_ratio
                losses = - torch.min(objective, clipped_objective)
                clip_ratios = objective > clipped_objective

                if self.config.tis_coef > 0:
                    # https://fengyao.notion.site/off-policy-rl
                    tis = torch.exp(
                        minibatch["logps"].detach() - minibatch["llm_logps"]
                    ).clamp(max=self.config.tis_coef)
                    losses *= tis

                loss, clip_ratio, entropy = aggregate_values(
                    (losses, clip_ratios, minibatch["entropy"]),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )
                loss = loss - self.config.entropy.coef * entropy
                if self.config.kl.coef > 0 and self.config.kl.type == "loss":
                    kl_loss = compute_approx_kl(
                        minibatch["logps"],
                        minibatch["ref_logps"],
                        self.config.kl.loss_estimator
                    ).sum() / total_actions
                    loss = loss + self.config.kl.coef * kl_loss

                metric = {
                    "actor/entropy": [entropy.item()],
                    "actor/loss": [loss.item()],
                    "actor/clip_ratio": [clip_ratio.item()],
                }

                return self.scale_loss(loss), 1, metric
            
            metric, grad_norm = self.forward_backward(f, batch)
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, mpu.get_data_parallel_group())
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        self.offload_model_to_cpu()

    @time_logger("update_rollout")
    def update_rollout(self, rollout, step):

        self.load_model_to_gpu()
        named_tensor_generator = self.bridge.export_weights(self.model)
        rollout.update(named_tensor_generator)
        self.offload_model_to_cpu()