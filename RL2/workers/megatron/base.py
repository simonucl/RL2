from functools import partial
from omegaconf import OmegaConf
from transformers import AutoConfig
from megatron.core import (
    parallel_state as mpu,
    tensor_parallel
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
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
                tensor_parallel_size=self.config.tp_size,
                expert_model_parallel_size=self.config.ep_size,
                expert_tensor_parallel_size=self.config.etp_size
            )
            tensor_parallel.model_parallel_cuda_manual_seed(42)

        self.device_mesh = {
            "pp": mpu.get_pipeline_model_parallel_group(),
            "dp": mpu.get_data_parallel_group(),
            "cp": mpu.get_context_parallel_group(),
            "tp": mpu.get_tensor_model_parallel_group()
        }

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