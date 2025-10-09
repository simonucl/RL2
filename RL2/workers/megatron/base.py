from omegaconf import OmegaConf
from transformers import AutoConfig
from megatron.core import (
    parallel_state as mpu,
    tensor_parallel
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from mbridge import AutoBridge
from ..base import Worker

class MegatronWorker(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        config = AutoConfig.from_pretrained(config.model_name)
        self.bridge = AutoBridge.from_config(config)
        if train and config.enable_gradient_checkpointing:
            gradient_checkpointing_config = OmegaConf.to_container(
                self.config.gradient_checkpointing_config
            )
            self.bridge.set_extra_args(**gradient_checkpointing_config)
        
    def prepare_device_mesh(self):

        if not mpu.is_initialized():
            mpu.initialize_model_parallel(
                pipeline_model_parallel_size=self.config.pp_size,
                context_parallel_size=self.config.sp_size,
                tensor_parallel_size=self.config.tp_size,
                expert_model_parallel_size=self.config.ep_size,
                expert_tensor_parallel_size=self.config.etp_size
            )
            tensor_parallel.model_parallel_cuda_manual_seed(42)

    def prepare_device_mesh(self):
        
        self.bridge.load_weights(self.model, self.config.model_name)

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            optimizer_config = OptimizerConfig(
                use_distributed_optimizer=True,
                **optimizer_config
            )
            self.optimizer = get_megatron_optimizer(
                optimizer_config,
                self.model
            )

        # TODO: offload to CPU