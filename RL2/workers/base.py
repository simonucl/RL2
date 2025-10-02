import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from transformers import AutoTokenizer
from RL2.utils.data_parallelism import prepare_dp_model
from RL2.utils.tensor_parallelism import prepare_tp_model
from RL2.utils.offloading import (
    load_model_to_device,
    optimizer_offloading_manager
)

class Worker:

    def __init__(self, config, train: bool):

        self.config = config
        self.train = train

        self.prepare_device_mesh()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        # Parse LoRA configuration
        self.lora_config = getattr(config, 'lora', None)
        self.use_lora = self.lora_config and getattr(self.lora_config, 'use_lora', False)

        # Validate LoRA compatibility
        if self.use_lora:
            if self.config.tp_size > 1:
                raise NotImplementedError(
                    "LoRA is currently incompatible with Tensor Parallelism (tp_size > 1). "
                    "This is due to PyTorch DTensor not supporting PEFT LoRA modules. "
                    "Please use tp_size=1 with LoRA, or use FSDP/DDP for parallelism."
                )
            if self.lora_config.r < 1:
                raise ValueError(f"LoRA rank must be >= 1, got {self.lora_config.r}")

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

        if self.train and self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Apply LoRA BEFORE any parallelism wrappers
        if self.use_lora and self.train:
            from peft import LoraConfig, get_peft_model, TaskType

            # Determine task type based on model class
            if hasattr(self.model, 'lm_head'):
                task_type = TaskType.CAUSAL_LM
            elif hasattr(self.model, 'score'):
                task_type = TaskType.SEQ_CLS
            else:
                task_type = TaskType.CAUSAL_LM  # default

            peft_config = LoraConfig(
                task_type=task_type,
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=list(self.lora_config.target_modules),
                bias="none",
                use_rslora=getattr(self.lora_config, 'use_rslora', False),
                modules_to_save=getattr(self.lora_config, 'modules_to_save', None)
            )
            self.model = get_peft_model(self.model, peft_config)

            if dist.get_rank() == 0:
                self.model.print_trainable_parameters()

        if self.config.tp_size > 1:
            prepare_tp_model(self.model, self.model_device_mesh["tp"])

        self.model = prepare_dp_model(
            self.model, self.model_device_mesh
        )

        if self.train:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )

        load_model_to_device(self, "cpu")
            
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