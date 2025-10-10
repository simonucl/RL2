from functools import partial
import torch
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from RL2.workers.megatron import MegatronWorker, forward_step
from RL2.utils.sequences import data_manager, count_total
from RL2.utils.megatron.context_parallelism import gather_along_cp
from RL2.utils.functions import (
    compute_logps_and_entropy, aggregate_values
)
from RL2.utils.logging import time_logger, gather_and_log


class MegatronActor(MegatronWorker):
    
    def __init__(self, config, train: bool):
        super().__init__(config, train)

        self.model = self.bridge.get_model(wrap_with_ddp=True)
        self.prepare_model_optimizer()

    @time_logger("compute_logps")
    @torch.no_grad()
    @data_manager(gather=True)
    def compute_logps(self, minibatches, step):
        pass

    @time_logger("update_actor")
    @data_manager() # TODO: boardcast along pp
    def sft_update(self, minibatches, step):

        total_actions, total_sequences = count_total(
            minibatches,
            ("action_mask", "eos_mask"),
            self.device_mesh["dp"]
        )

        def loss_func(minibatch, logits):

            minibatch["logps"] = compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch)
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            return (
                mpu.get_data_parallel_world_size(with_context_parallel=True) * loss,
                1.0,
                {"loss": loss.item()}
            )

        forward_backward_func = get_forward_backward_func()
        metrics = forward_backward_func(
            model=self.model,
            data_iterator=iter(minibatches),
            num_microbatches=len(minibatches),
            forward_step_func=partial(forward_step, loss_func),
            seq_length=1,
            micro_batch_size=1,
            forward_only=False # optional
        )
        # TODO: abstract optimizer_step
        self.optimizer.step()
        self.optimizer.zero_grad()
        if mpu.is_pipeline_last_stage():
            metrics = {
                k: [metric[k] for metric in metrics]
                for k in metrics.keys()
            }
            gather_and_log(
                metrics,
                step,
                mpu.get_data_model_parallel_group()
            )