from collections import defaultdict
import torch
from megatron.core import parallel_state as mpu
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.algorithms import dpo_loss, actor_ppo_loss
from RL2.utils.logging import (
    time_logger,
    gather_and_log,
    gather_and_reduce,
    rank0_log
)


class MegatronActor(MegatronWorker):
    
    def __init__(self, config, train: bool):
        super().__init__(config, train)

        self.model = self.provider.provide_distributed_model(
            ddp_config=self.ddp_config,
            wrap_with_ddp=train
        )
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
            return self.scale_loss(loss), {"loss": [loss.item()]}

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
            losses, metric = dpo_loss(self.config, minibatch)
            loss = losses.sum() / total_pairs
            metric["loss"] = [loss.item()]
            return self.scale_loss(loss), metric

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

                losses, clip_ratios = actor_ppo_loss(self.config, minibatch)

                loss, clip_ratio, entropy = aggregate_values(
                    (losses, clip_ratios, minibatch["entropy"]),
                    minibatch["action_mask"],
                    self.config.avg_level,
                    total_actions,
                    total_sequences
                )

                metric = {
                    "actor/entropy": [entropy.item()],
                    "actor/loss": [loss.item()],
                    "actor/clip_ratio": [clip_ratio.item()],
                }

                return self.scale_loss(loss), metric
            
            metric, grad_norm = self.forward_backward(f, batch)
            for k, v in metric.items():
                metrics[k].append(
                    gather_and_reduce(v, mpu.get_data_parallel_group())
                )
            metrics["actor/grad_norm"].append(grad_norm)

        rank0_log(metrics, step)
        if self.config.adv_estimator == "gae":
            self.offload_model_to_cpu()

    @time_logger("update_rollout")
    def update_rollout(self, rollout, step):

        self.load_model_to_gpu()
        named_tensor_generator = self.bridge.export_hf_weights(
            self.model, cpu=True
        )
        rollout.update(named_tensor_generator)
        self.offload_model_to_cpu()