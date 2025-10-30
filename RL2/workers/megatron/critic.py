from collections import defaultdict
import torch
from megatron.core import parallel_state as mpu
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
from RL2.utils.functions import aggregate_values
from RL2.utils.algorithms import rm_loss, critic_ppo_loss
from RL2.utils.logging import (
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class MegatronCritic(MegatronWorker):
    
    def __init__(self, config):
        super().__init__(config, True)

        self.model = self.provider.provide_distributed_model(
            ddp_config=self.ddp_config,
            wrap_with_ddp=True
        ) # TODO: make value model
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
            losses, metric = rm_loss(minibatch)
            loss = losses.sum() / total_pairs
            metric["loss"] = [loss.item()]
            return self.scale_loss(loss), metric

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
                losses, clip_ratios = critic_ppo_loss(self.config, minibatch)

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

                return self.scale_loss(loss), metric

            metric, grad_norm = self.forward_backward(f, batch)
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, mpu.get_data_parallel_group())
                )
            metrics["critic/grad_norm"].append(grad_norm)
        
        rank0_log(metrics, step)
        self.offload_model_to_cpu()