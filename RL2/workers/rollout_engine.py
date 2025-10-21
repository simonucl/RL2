from omegaconf import OmegaConf
import asyncio
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
import weave
import gc
import os
from RL2.datasets import pack_tensor_dicts
from RL2.utils.sglang import (
    prepare_environment_variables,
)
from RL2.utils.logging import time_logger, gather_and_log
from RL2.workers.rollout import Rollout
from RL2.utils.communication import gather_and_concat_list, split_and_scatter_list

def _postprocess_generate(x):
    return {
        'text': x['text'],
        'meta_info': {k: v for k, v in x['meta_info'].items() if 'token_logprobs' not in k}
    }
    
class RolloutEngine(Rollout):

    def __init__(self, config):

        self.config = config
        self.lora_loaded = False
        self.prepare_device_mesh()
        self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
        prepare_environment_variables(self.device_mesh["tp"].get_group())
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.prepare_environment()
        
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            self.llm = Engine(
                enable_memory_saver=True,
                model_path=config.server_args.model_path,
                dtype=config.server_args.dtype,
                tp_size=config.server_args.tp_size,
                mem_fraction_static=config.server_args.mem_fraction_static,
                enable_lora=config.server_args.enable_lora,
                max_lora_rank=config.server_args.max_lora_rank,
                lora_target_modules=config.server_args.lora_target_modules,
            )
            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    @weave.op(postprocess_output=_postprocess_generate)
    async def async_generate(self, states, sampling_params):
        return await self.llm.async_generate(
            input_ids=states,
            sampling_params=sampling_params,
            return_logprob=True
        )
        
    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # The data is distributed from rank 0 before each worker operation
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            data_list = split_and_scatter_list(
                data_list, self.device_mesh["dp"]
            )
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(ex, train) for ex in data_list),
                    desc="Rollout",
                    position=1,
                    leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )
            if train:
                # If test, llm will soon be called again. See `Trainer.train`.
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:

            all_tensor_dicts, metrics = map(list, zip(*outputs))

            suffix = "train" if train else "test"
            metrics = {
                f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            gather_and_log(metrics, step, self.device_mesh["dp"].get_group())

            if not train:
                return

            all_tensor_dicts = gather_and_concat_list(
                all_tensor_dicts, self.device_mesh["dp"].get_group()
            )

            if dist.get_rank() == 0:

                group_size = self.config.responses_per_prompt
                if group_size > 1 and self.config.dynamic_filtering:

                    rewards = torch.FloatTensor([
                        sum([td["rewards"].sum().item() for td in tensor_dicts])
                        for tensor_dicts in all_tensor_dicts
                    ]).view(-1, group_size)
                    are_filtered = rewards.std(-1) == 0
                    all_tensor_dicts = sum([
                        all_tensor_dicts[idx * group_size:(idx + 1) * group_size]
                        for idx, is_filtered in enumerate(are_filtered)
                        if not is_filtered
                    ], [])
                    wandb.log({
                        "dynamic_filtering_ratio": are_filtered.float().mean().item()
                    }, step=step)

                tensor_dicts = sum(all_tensor_dicts, [])
                tensor_dict = pack_tensor_dicts(tensor_dicts)
                seqs = torch.LongTensor([
                    len(tensor_dicts) for tensor_dicts in all_tensor_dicts
                ])
                cu_seqs = torch.cumsum(
                    torch.cat((torch.LongTensor([0]), seqs)), dim=0
                )

                return tensor_dict, cu_seqs

        return None, None

    @torch.no_grad()
    def update_lora(self, lora_dir):
        self.llm.flush_cache()
        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        self.llm.resume_memory_occupation()

        if self.device_mesh["tp"].get_local_rank() == 0:
            if self.lora_loaded:
                self.llm.unload_lora_adapter(lora_name="default")
                self.lora_loaded = False
            self.llm.load_lora_adapter(lora_path=lora_dir)
            self.lora_loaded = True

        dist.barrier()

    @torch.no_grad()
    def update(self, named_tensor_generator):

        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.flush_cache()
        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        dist.barrier()

        for idx, (name, tensor) in enumerate(named_tensor_generator):
            tensor = tensor.to(torch.cuda.current_device())
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            serialized_tensors = [
                None for _ in range(self.device_mesh["tp"].size())
            ] if self.device_mesh["tp"].get_local_rank() == 0 else None
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(name, LocalSerializedTensor(values=serialized_tensors))],
                    flush_cache=(idx == len(named_tensor_generator) - 1)
                )

        del named_tensor_generator
        gc.collect()
        torch.cuda.empty_cache()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            self.llm.flush_cache()
        dist.barrier()