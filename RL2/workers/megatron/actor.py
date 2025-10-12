from functools import partial
import torch
from megatron.core import parallel_state as mpu
from RL2.workers.megatron import MegatronWorker
from RL2.utils.sequences import count_total, gather_along_cp
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
        def f(minibatch, cu_seqlens, logits, non_loss_data=True):

            compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, cu_seqlens)
            return minibatch
        
        minibatches = self.forward_backward(f, minibatches, step, False)

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

        def f(minibatch, cu_seqlens, logits):

            compute_logps_and_entropy(
                logits,
                minibatch,
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, cu_seqlens)
            loss = aggregate_values(
                - minibatch["logps"],
                minibatch["action_mask"],
                self.config.avg_level,
                total_actions,
                total_sequences
            )
            return self.scale_loss(loss), 1, {"loss": loss.item()}

        self.forward_backward(f, minibatches, step)

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
                mpu.get_tensor_model_parallel_group(),
                return_entropy=False
            )
            minibatch = gather_along_cp(minibatch, cu_seqlens)
            loss, metric = dpo_loss(
                minibatch["logps"],
                minibatch["ref_logps"],
                self.config.beta
            )
            return self.scale_loss(loss.sum() / total_pairs), 1, metric

        self.forward_backward(f, minibatches, step)

    @time_logger("update_actor")
    def ppo_update(self, tensor_dict, step):
        if step < self.config.freeze_steps:
            return
        batches = self.scatter_data(tensor_dict, pack_minibatches=True)
        self.load_model_to_device(torch.cuda.current_device())

        def f(minibatch, cu_seqlens, logits):
            pass

        self.model.train()
        for batch in batches:

            total_actions, total_sequences = count_total(
                batch,
                ("action_mask", "eos_mask"),
                mpu.get_data_parallel_group()
            )
            
            self.forward_backward(f, batch, step)