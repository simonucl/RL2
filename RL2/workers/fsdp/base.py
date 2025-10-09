from omegaconf import OmegaConf
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from transformers import get_scheduler
from ..base import Worker
from RL2.utils.fsdp.data_parallelism import prepare_dp_model
from RL2.utils.fsdp.tensor_parallelism import prepare_tp_model
from RL2.utils.fsdp.offloading import (
    load_model_to_device,
    optimizer_offloading_manager
)

class FSDPWorker(Worker):

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % (self.config.ddp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by ddp_size {self.config.ddp_size} * tp_size {self.config.tp_size}."
        self.fsdp_size = world_size // (self.config.ddp_size * self.config.tp_size)
        self.model_device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("ddp", "fsdp", "tp"),
            mesh_shape=(self.config.ddp_size, self.fsdp_size, self.config.tp_size)
        )

        assert world_size % (self.config.sp_size * self.config.tp_size) == 0, \
            f"World_size {world_size} must be divisible by sp_size {self.config.sp_size} * tp_size {self.config.tp_size}."
        self.dp_size = world_size // (self.config.sp_size * self.config.tp_size)
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_dim_names=("dp", "sp", "tp"),
            mesh_shape=(self.dp_size, self.config.sp_size, self.config.tp_size)
        )

    def prepare_model_optimizer(self):

        if self.train and self.config.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model, self.model_device_mesh
        )

        if self.train:

            optimizer_config = OmegaConf.to_container(self.config.optimizer)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_config
            )

        load_model_to_device(self, "cpu")
    
    def prepare_scheduler(self, total_steps):

        num_training_steps = total_steps * getattr(
            self.config, "update_per_rollout", 1
        )
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        self.scheduler = get_scheduler(
            self.config.scheduler,
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def backward(self, loss):
        # https://github.com/ChenmienTan/RL2/issues/11
        (self.dp_size * self.config.sp_size * loss).backward()
    
    @optimizer_offloading_manager
    def optimizer_step(self):

        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return grad_norm.item()