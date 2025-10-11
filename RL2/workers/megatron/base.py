from functools import partial
from omegaconf import OmegaConf
from transformers import AutoConfig
from megatron.core import (
    parallel_state as mpu,
    tensor_parallel
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from mbridge import AutoBridge
from RL2.workers import Worker
from RL2.utils.megatron.context_parallelism import slide_along_cp

def forward_step(f, data_iterator, model):

    minibatch = next(data_iterator)
    minibatch, packed_seq_params = slide_along_cp(minibatch)
    output_tensor = model(
        input_ids=minibatch["states"],
        attention_mask=None,
        position_ids=None,
        labels=None,
        packed_seq_params=packed_seq_params
    )

    return output_tensor, partial(f, minibatch, packed_seq_params)


class MegatronWorker(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        config = AutoConfig.from_pretrained(config.model_name)
        self.bridge = AutoBridge.from_config(config)
        tf_config = OmegaConf.to_container(self.config.tf_config)
        self.bridge.set_extra_args(**tf_config)

    def prepare_device_mesh(self):

        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                pipeline_model_parallel_size=self.config.pp_size,
                context_parallel_size=self.config.cp_size,
                tensor_model_parallel_size=self.config.tp_size,
                expert_model_parallel_size=self.config.ep_size,
                expert_tensor_parallel_size=self.config.etp_size
            )
            tensor_parallel.model_parallel_cuda_manual_seed(42)

    def prepare_model_optimizer(self):
        
        self.bridge.load_weights(self.model, self.config.model_name)

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            optimizer_config = OptimizerConfig(
                use_distributed_optimizer=True,
                **optimizer_config
            )
            self.optimizer = get_megatron_optimizer(
                optimizer_config, self.model
            )

    def prepare_scheduler(self, total_steps):

        num_training_steps = total_steps * getattr(
            self.config, "update_per_rollout", 1
        )
        lr_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        lr_decay_steps = num_training_steps - lr_warmup_steps

        return OptimizerParamScheduler(
            self.optimizer,
            init_lr=0.0,
            max_lr=self.config.optimizer.lr,
            min_lr=0.0,
            lr_warmup_steps=lr_warmup_steps,
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=self.config.scheduler,
            start_wd=self.config.optimizer.weight_decay,
            end_wd=self.config.optimizer.weight_decay,
            wd_incr_steps=0,
            wd_incr_style="constant"
        )