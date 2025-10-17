from collections import defaultdict
import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from mbridge.utils.post_creation_callbacks import make_value_model
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
from RL2.utils.functions import aggregate_values
from RL2.utils.logging import (
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)

class MegatronCritic(MegatronWorker):
    
    def __init__(self, config):
        super().__init__(config, True)

        self.model = self.bridge.get_model(
            wrap_with_ddp=True,
            post_model_creation_callbacks=[
                make_value_model
            ]
        )
        self.prepare_model_optimizer()

    @time_logger("compute_values")
    @torch.no_grad()
    def compute_values(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)
        self.load_model_to_gpu()

        for model in self.model:
            model.eval()
        def f(minibatch, cu_seqlens, logits, non_loss_data=True):

            minibatch["old_values"] = logits.squeeze(-1) * minibatch["action_mask"]
            return gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
        
        minibatches = self.forward_backward(f, minibatches)

        self.offload_model_to_cpu()
        return self.gather_data(minibatches)

    @time_logger("update_critic")
    def rm_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict, pair=True)

        total_pairs = count_total(
            minibatches, "eos_mask", mpu.get_data_parallel_group()
        ) // 2
        
        def f(minibatch, cu_seqlens, logits):

            minibatch["values"] = logits.squeeze(-1) * minibatch["action_mask"]
            minibatch = gather_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                cu_seqlens
            )
            chosen_rewards, rejected_rewards = minibatch["values"].sum(-1).view(-1, 2).T
            reward_margins = chosen_rewards - rejected_rewards
            loss = - F.logsigmoid(reward_margins).sum() / total_pairs
            metric = {
                "loss": [loss.item()],
                "accuracy": (reward_margins > 0).tolist()
            }
            return self.scale_loss(loss), 1, metric

        metrics, grad_norm = self.forward_backward(f, minibatches)
        metrics["grad_norm"] = [grad_norm]
        gather_and_log(metrics, step, mpu.get_data_parallel_group())

    @time_logger("update_critic")
    def ppo_update(self, tensor_dict, step):
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

                minibatch["values"] = logits.squeeze(-1) * minibatch["action_mask"]
                minibatch = gather_along_cp(
                    minibatch,
                    mpu.get_context_parallel_group(),
                    cu_seqlens
                )
                clipped_values = torch.clamp(
                    minibatch["values"],
                    minibatch["old_values"] - self.config.clip,
                    minibatch["old_values"] + self.config.clip
                )

                mse = (minibatch["values"] - minibatch["returns"]).pow(2)
                clipped_mse = (clipped_values - minibatch["returns"]).pow(2)
                losses = torch.max(mse, clipped_mse)
                clip_ratios = mse < clipped_mse

                loss, clip_ratio = aggregate_values(
                    (losses, clip_ratios),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )

                metric = {
                    "critic/loss": [loss.item()],
                    "critic/clip_ratio": [clip_ratio.item()]
                }

                return self.scale_loss(loss), 1, metric

            metric, grad_norm = self.forward_backward(f, batch)
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, mpu.get_data_parallel_group())
                )
            metrics["critic/grad_norm"].append(grad_norm)
        
        rank0_log(metrics, step)
        self.offload_model_to_cpu()