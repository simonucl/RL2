from functools import partial
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from transformers import AutoConfig
from megatron.core import (
    parallel_state as mpu,
    tensor_parallel,
    dist_checkpointing
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.dist_checkpointing.serialization import (
        get_default_load_sharded_strategy,
        get_default_save_sharded_strategy
    )
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper
)
from mbridge import AutoBridge
from RL2.workers import Worker
from RL2.utils.sequences import scatter_data, gather_data, slide_along_cp
from RL2.utils.logging import gather_and_log


class MegatronWorker(Worker):

    def __init__(self, config, train: bool):
        super().__init__(config, train)
        
        config = AutoConfig.from_pretrained(config.model_name)
        self.bridge = AutoBridge.from_config(config)
        tf_config = (
            OmegaConf.to_container(self.config.tf_config)
            if hasattr(self.config, "tf_config") else {}
        )
        self.bridge.set_extra_args(
            bf16=True,
            attention_backend="flash",
            **tf_config
        )

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
                bf16=True,
                params_dtype=torch.bfloat16,
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

        self.scheduler = OptimizerParamScheduler(
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

    def scatter_data(
        self,
        tensor_dict,
        pack_minibatches: bool = False,
        pair: bool = False
    ):
        max_length_per_dp = mpu.get_context_parallel_world_size() * mpu.get_tensor_model_parallel_world_size() * (
            self.config.max_length_per_device
            if torch.is_grad_enabled()
            else self.config.max_inference_length_per_device
        )
        return scatter_data(
            tensor_dict,
            mpu.get_data_parallel_group(),
            max_length_per_dp,
            self.config.update_per_rollout if pack_minibatches else 1,
            pair
        )

    def gather_data(self, minibatches):
        return gather_data(minibatches, mpu.get_data_parallel_group())

    def scale_loss(self, loss):
        return mpu.get_data_parallel_world_size(with_context_parallel=True) * loss

    def forward_backward(self, f, minibatches, step):

        def forward_step(data_iterator, model):

            minibatch = next(data_iterator)
            minibatch, cu_seqlens = slide_along_cp(
                minibatch,
                mpu.get_context_parallel_group(),
                mpu.get_tensor_model_parallel_world_size()
            )
            global_cu_seqlens = mpu.get_context_parallel_world_size() * cu_seqlens
            max_seqlen = (global_cu_seqlens[1:] - global_cu_seqlens[:-1]).max().item()
            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=global_cu_seqlens,
                cu_seqlens_kv=global_cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_kv=max_seqlen,
                qkv_format="thd"
            )
            output_tensor = model(
                input_ids=minibatch["states"],
                attention_mask=None,
                position_ids=None,
                labels=None,
                packed_seq_params=packed_seq_params
            )

            return output_tensor, partial(f, minibatch, cu_seqlens)

        forward_backward = get_forward_backward_func()
        output = forward_backward(
            model=self.model,
            data_iterator=iter(minibatches),
            num_microbatches=len(minibatches),
            forward_step_func=forward_step,
            seq_length=1,
            micro_batch_size=1,
            forward_only=not torch.is_grad_enabled()
        )
        if torch.is_grad_enabled():
            _, grad_norm, _ = self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step(1)
            if mpu.is_pipeline_last_stage():
                metrics = {
                    k: sum([metric[k] for metric in output], [])
                    for k in output[0].keys()
                }
                metrics["grad_norm"] = [grad_norm]
                gather_and_log(metrics, step, mpu.get_data_parallel_group())
        else:
            return output

    def get_ckpt(self):

        ckpt = {"model": self.model[0].sharded_state_dict()}
        ckpt["optimizer"] = self.optimizer.shardedstate_dict(ckpt)
        ckpt["scheduler"] = self.scheduler.state_dict()
        return ckpt

    def load_ckpt(self, save_dir):
        
        ckpt = self.get_ckpt()
        sharded_strategy = get_default_load_sharded_strategy(save_dir)
        sharded_strategy = FullyParallelLoadStrategyWrapper(
            sharded_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True)
        )
        ckpt = dist_checkpointing.load(
            ckpt,
            save_dir,
            sharded_strategy=sharded_strategy
        )
        self.model[0].load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

    def save_ckpt(self, save_dir):

        sharded_strategy = get_default_save_sharded_strategy("torch_dist")
        sharded_strategy = FullyParallelSaveStrategyWrapper(
            sharded_strategy,
            mpu.get_data_parallel_group(with_context_parallel=True)
        )
        dist_checkpointing.save(
            self.get_ckpt(),
            save_dir,
            sharded_strategy=sharded_strategy
        )

    def save_model(self, save_dir):

        self.bridge.save_weights(self.model, save_dir)
        if dist.get_rank() == 0:
            self.tokenizer.save_pretrained(save_dir)
        dist.barrier()