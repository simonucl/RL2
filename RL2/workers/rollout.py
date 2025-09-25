from omegaconf import OmegaConf
import base64
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.datasets import (
    initialize_state_dict,
    state_dict_to_tensor_dict,
    pack_tensor_dicts
)
from RL2.utils.sglang import (
    prepare_environment_variables,
    launch_server_process,
    launch_router_process,
    make_request,
    async_generate
)
from RL2.utils.checkpointing import get_state_dict
from RL2.utils.logging import time_logger, gather_and_log


class Rollout:

    def __init__(self, config):
        
        self.config = config
        self.prepare_device_mesh()
        prepare_environment_variables(self.device_mesh["tp"])
        if self.device_mesh["tp"].get_local_rank() == 0:

            self.worker_url = launch_server_process(config.server_args)
            worker_urls = [
                None for _ in range(self.device_mesh["dp"].size())
            ] if self.device_mesh["dp"].get_local_rank() == 0 else None
            dist.gather_object(
                self.worker_url,
                worker_urls,
                group_dst=0,
                group=self.device_mesh["dp"].get_group(),
            )
        
        if dist.get_rank() == 0:

            self.prepare_environment()
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.server_args.model_path, trust_remote_code=True
            )
            self.router_url = launch_router_process(worker_urls)

            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % self.config.server_args.tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {self.config.server_args.tp_size}."
        self.dp_size = world_size // self.config.server_args.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.server_args.tp_size)
        )

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)
        
    async def rollout(self, data, train):

        if "prompt" in data:
            state_text = data["prompt"]
        else:
            state_text, data["extra_info"] = await self.env.reset(
                data["extra_info"]
            )
        state_dict = initialize_state_dict(self.tokenizer, state_text)
        env_response = {"extra_info": data["extra_info"]}
        tensor_dicts = []
        metric = defaultdict(list)
        scores = []
        for turn in range(1, self.config.max_turns + 1):

            llm_response = await async_generate(
                self.router_url,
                state_dict["states"],
                self.train_sampling_params
                if train else self.test_sampling_params
            )

            action_text = llm_response["text"]
            env_response = await self.env.step(
                state_text, action_text, env_response["extra_info"]
            )

            meta_info = llm_response["meta_info"]
            logp, action, _ = map(list, zip(*meta_info["output_token_logprobs"]))
            state_dict["states"].extend(action)
            state_dict["actions"].extend(action)
            state_dict["action_mask"].extend(len(action) * [1])
            state_dict["logps"].extend(logp)
            state_dict["rewards"].extend((len(action) - 1) * [0] + [env_response["reward"]])
            metric["response_length"].append(meta_info["completion_tokens"])
            metric["length_clip_ratio"].append(
                meta_info["finish_reason"]["type"] == "length"
            )
            scores.append(env_response["score"])

            if turn == self.config.max_turns or env_response["done"]:
                tensor_dicts.append(state_dict_to_tensor_dict(state_dict))
                break
            if env_response["next_state"].startswith(state_text + action_text):
                state_dict_delta = initialize_state_dict(
                    self.tokenizer,
                    env_response["next_state"][len(state_text + action_text):]
                )
                for k, v in state_dict_delta.items():
                    state_dict[k].extend(v)
            else:
                tensor_dicts.append(state_dict_to_tensor_dict(state_dict))
                state_dict = initialize_state_dict(
                    self.tokenizer,
                    env_response["next_state"]
                )
            state_text = env_response["next_state"]

        metric["n_turns"].append(turn)
        metric["scores"].append(sum(scores))

        return tensor_dicts, metric

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        if dist.get_rank() == 0:

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(data, train) for data in data_list),
                    desc="Rollout",
                    position=1,
                    leave=False,
                    disable=(dist.get_rank() != 0)
                )
            )

            all_tensor_dicts, metrics = map(list, zip(*outputs))
            suffix = "train" if train else "test"
            metrics = {
                f"{k}/{suffix}": sum([metric[k] for metric in metrics], [])
                for k in metrics[0].keys()
            }
            gather_and_log(metrics, step)

        dist.barrier()

        if not train:
            return

        if self.device_mesh["tp"].get_local_rank() == 0:
            make_request(
                self.worker_url, "release_memory_occupation"
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
        
    @time_logger("update_rollout")
    def update(self, actor, step):

        state_dict = get_state_dict(actor)
        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        if self.device_mesh["tp"].get_local_rank() == 0:
            make_request(
                self.worker_url, "resume_memory_occupation"
            )
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
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
                named_tensors = [
                    (name, LocalSerializedTensor(values=serialized_tensors))
                ]
                serialized_named_tensors = [
                    MultiprocessingSerializer.serialize(named_tensors)
                    for _ in range(self.device_mesh["tp"].size())
                ]
                serialized_named_tensors = [
                    base64.b64encode(snt).decode("utf-8")
                    for snt in serialized_named_tensors
                ]
                payload = {
                    "serialized_named_tensors": serialized_named_tensors,
                    "flush_cache": (idx == len(state_dict) - 1)
                }
                make_request(
                    self.worker_url, "update_weights_from_tensor", payload
                )
        dist.barrier()