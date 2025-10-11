from functools import partial
import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from RL2.workers.megatron import MegatronWorker, forward_step
from RL2.utils.sequences import count_total
from RL2.utils.megatron.context_parallelism import gather_along_cp
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.algorithms import (
    dpo_loss
)
from RL2.utils.logging import time_logger, gather_and_log


class MegatronActor(MegatronWorker):
    
    def __init__(self, config, train: bool):
        super().__init__(config, train)

        self.model = self.bridge.get_model(wrap_with_ddp=True)
        self.prepare_model_optimizer()

    @time_logger("compute_logps")
    @torch.no_grad()
    def compute_logps(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)
        self.load_model_to_device(torch.cuda.current_device())

        prefix = "old" if self.train else "ref"
        self.model.eval()
        def f(minibatch, packed_seq_lens, logits, non_loss_data=True):

            minibatch[f"{prefix}_logps"] = compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, packed_seq_lens)
            return minibatch
        
        forward_backward_func = get_forward_backward_func()
        minibatches = forward_backward_func(
            model=self.model,
            data_iterator=iter(minibatches),
            num_microbatches=len(minibatches),
            forward_step_func=partial(forward_step, f),
            seq_length=1,
            micro_batch_size=1,
            forward_only=True
        )

        self.load_model_to_device("cpu")
        return self.gather_data(minibatches)

    @time_logger("update_actor")
    def sft_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict)

        total_actions, total_sequences = count_total(
            minibatches,
            ("action_mask", "eos_mask"),
            mpu.get_data_parallel_group()
        )

        def f(minibatch, packed_seq_lens, logits):

            minibatch["logps"] = compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, packed_seq_lens)
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            return self.scale_loss(loss), 1, {"loss": loss.item()}

        forward_backward_func = get_forward_backward_func()
        metrics = forward_backward_func(
            model=self.model,
            data_iterator=iter(minibatches),
            num_microbatches=len(minibatches),
            forward_step_func=partial(forward_step, f),
            seq_length=1,
            micro_batch_size=1
        )
        grad_norm = self.optimizer_step()
        if mpu.is_pipeline_last_stage():
            metrics = {
                k: [metric[k] for metric in metrics]
                for k in metrics[0].keys()
            }
            metrics["grad_norm"] = [grad_norm]
            gather_and_log(metrics, step, mpu.get_data_parallel_group())

    @time_logger("update_actor")
    def dpo_update(self, tensor_dict, step):
        minibatches = self.scatter_data(tensor_dict, pair=True)

        total_pairs = count_total(
            minibatches, "eos_mask", mpu.get_data_parallel_group()
        ) // 2

        def f(minibatch, packed_seq_lens, logits):

            minibatch["logps"] = compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, packed_seq_lens)
            loss, metric = dpo_loss(
                minibatch["logps"],
                minibatch["ref_logps"],
                self.config.beta
            )
            return self.scale_loss(loss.sum() / total_pairs), 1, metric

        forward_backward_func = get_forward_backward_func()
        metrics = forward_backward_func(
            model=self.model,
            data_iterator=iter(minibatches),
            num_microbatches=len(minibatches),
            forward_step_func=partial(forward_step, f),
            seq_length=1,
            micro_batch_size=1
        )
        grad_norm = self.optimizer_step()
        if mpu.is_pipeline_last_stage():
            metrics = {
                k: [metric[k] for metric in metrics]
                for k in metrics[0].keys()
            }
            metrics["grad_norm"] = [grad_norm]
            gather_and_log(metrics, step, mpu.get_data_parallel_group())