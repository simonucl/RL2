from omegaconf import OmegaConf
import time
import base64
import asyncio
import aiohttp
import requests
import importlib
import requests
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from RL2.datasets import get_tensor_dict, pack_tensor_dicts
from RL2.utils.sglang import (
    prepare_environment_variables,
    launch_server_process,
    launch_router_process
)
from RL2.utils.logging import time_logger, gather_and_log


class Rollout:

    def __init__(self, config):
        
        self.config = config
        self.prepare_device_mesh()
        prepare_environment_variables(self.device_mesh["tp"].get_group())
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
            
            # Validate worker URLs before launching router
            valid_worker_urls = []
            for url in worker_urls:
                if url is not None:
                    try:
                        response = requests.get(f"{url}/health_generate", timeout=10)
                        if response.status_code == 200:
                            valid_worker_urls.append(url)
                            print(f"✅ Worker {url} is healthy")
                        else:
                            print(f"❌ Worker {url} returned status {response.status_code}")
                    except Exception as e:
                        print(f"❌ Worker {url} is unreachable: {e}")
            
            if not valid_worker_urls:
                raise RuntimeError("No valid worker URLs found!")
            
            print(f"Using {len(valid_worker_urls)}/{len([u for u in worker_urls if u is not None])} workers")
            self.router_url = launch_router_process(valid_worker_urls)

            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

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

    def make_request(self, endpoint, method="POST", payload=None):

        if self.device_mesh["tp"].get_local_rank() == 0:
            while True:
                try:
                    if method == "POST":
                        response = requests.post(
                            f"{self.worker_url}/{endpoint}",
                            json=payload or {}
                        )
                    elif method == "GET":
                        response = requests.get(
                            f"{self.worker_url}/{endpoint}"
                        )
                    else:
                        raise NotImplementedError
                    response.raise_for_status()
                    return
                except NotImplementedError:
                    raise
                except:
                    time.sleep(1)

    async def async_generate(self, states, sampling_params):
        
        payload = {
            "input_ids": states,
            "sampling_params": sampling_params,
            "return_logprob": True
        }

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.post(
                        f"{self.router_url}/generate",
                        json=payload
                    ) as response:
                        return await response.json(content_type=None)
                except:
                    await asyncio.sleep(1)
        
    async def rollout(self, data, train):

        def initialize_state_dict(state_text):
            states = self.tokenizer.encode(state_text, add_special_tokens=False)
            return {
                "states": states,
                "actions": len(states) * [0],
                "action_mask": len(states) * [0],
                "logps": len(states) * [0],
                "rewards": len(states) * [0]
            }

        def state_dict_to_tensor_dict(state_dict):

            tensor_dict = get_tensor_dict(
                state_dict["states"],
                state_dict["actions"],
                state_dict["action_mask"]
            )
            tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
            tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
            return tensor_dict

        if "prompt" in data:
            state_text = data["prompt"]
        else:
            state_text, data["extra_info"] = await self.env.reset(
                data["extra_info"]
            )
        state_dict = initialize_state_dict(state_text)
        env_response = {"extra_info": data["extra_info"]}
        tensor_dicts = []
        metric = defaultdict(list)
        scores = []
        for turn in range(1, self.config.max_turns + 1):

            llm_response = await self.async_generate(
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
                    env_response["next_state"][len(state_text + action_text):]
                )
                for k, v in state_dict_delta.items():
                    state_dict[k].extend(v)
            else:
                tensor_dicts.append(state_dict_to_tensor_dict(state_dict))
                state_dict = initialize_state_dict(env_response["next_state"])
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

        self.make_request("release_memory_occupation")

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
    def update(self, named_tensor_generator):

        torch.cuda.empty_cache()
        dist.barrier()
        # or resume_memory_occupation() may OOM
        self.make_request("resume_memory_occupation")
        
        for name, tensor in named_tensor_generator:
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
                    "flush_cache": False
                }
                self.make_request("update_weights_from_tensor", payload=payload)
        self.make_request("flush_cache", "GET")