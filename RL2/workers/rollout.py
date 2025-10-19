from omegaconf import OmegaConf
import os
import base64
import asyncio
import importlib
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
import weave
from RL2.workers import Worker
from RL2.datasets import get_tensor_dict, pack_tensor_dicts
from RL2.utils.communication import split_and_scatter_list, gather_and_concat_list
from RL2.utils.logging import time_logger, gather_and_log

def postprocess_output(response):
    tensor_dicts, metrics, last_response = response
    return {
        **metrics,
        "response": last_response
    }

class Rollout(Worker):

    def __init__(self, config, actor_config=None):

        self.config = config
        self.actor_config = actor_config
        self.train = None

        # LoRA configuration from actor
        self.lora_enabled = actor_config and getattr(actor_config, 'use_lora', False)
        self.lora_rank = getattr(actor_config.lora, 'r', 64) if self.lora_enabled else 64
        self.lora_loaded = False

        self.prepare_device_mesh()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )

        self.prepare_environment_variables()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.prepare_environment()

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            engine_kwargs = {
                "model_path": config.model_name,
                "dtype": config.dtype,
                "tp_size": self.device_mesh["tp"].size(),
                "mem_fraction_static": config.mem_fraction_static,
                "enable_memory_saver": True,
            }

            if hasattr(config, 'context_length') and config.context_length:
                engine_kwargs['context_length'] = config.context_length

            if self.lora_enabled:
                engine_kwargs['enable_lora'] = True
                engine_kwargs['max_loras_per_batch'] = 1
                engine_kwargs['max_lora_rank'] = self.lora_rank
                if dist.get_rank() == 0:
                    print(f"[LoRA Init] Launching SGLang Engine with LoRA support (max_rank={self.lora_rank})")

            self.llm = Engine(**engine_kwargs)

        self.train_sampling_params = OmegaConf.to_container(
            config.train_sampling_params
        )
        self.test_sampling_params = OmegaConf.to_container(
            config.test_sampling_params
        )

    def prepare_device_mesh(self):

        world_size = dist.get_world_size()
        assert world_size % self.config.tp_size == 0, \
            f"World_size {world_size} must be divisible by tp_size {self.config.tp_size}."
        self.dp_size = world_size // self.config.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.tp_size)
        )

    def prepare_environment_variables(self):

        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            cuda_visible_devices = cuda_visible_devices.split(",")
            cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
        else:
            cuda_visible_device = os.environ["LOCAL_RANK"]
        cuda_visible_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            cuda_visible_devices,
            cuda_visible_device,
            self.device_mesh["tp"].get_group(),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def prepare_environment(self):

        spec = importlib.util.spec_from_file_location(
            "custom_module", self.config.env_path
        )
        self.env = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.env)

    def initialize_state_dict(self, state_text):

        states = self.tokenizer.encode(state_text, add_special_tokens=False)
        return {
            "states": states,
            "actions": len(states) * [0],
            "action_mask": len(states) * [0],
            "logps": len(states) * [0],
            "rewards": len(states) * [0]
        }

    def get_tensor_dict(self, state_dict):

        tensor_dict = get_tensor_dict(
            state_dict["states"],
            state_dict["actions"],
            state_dict["action_mask"]
        )
        tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
        tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
        return tensor_dict

    @weave.op(postprocess_output=postprocess_output)
    async def rollout(self, ex, train):

        state_text = (
            ex["prompt"] if "prompt" in ex else
            await self.env.reset(ex["extra_info"])
        )
        state_dict = self.initialize_state_dict(state_text)
        env_response = {"extra_info": ex["extra_info"]}
        tensor_dicts = []
        metric = defaultdict(list)
        scores = []
        for turn in range(1, self.config.max_turns + 1):

            llm_response = await self.llm.async_generate(
                input_ids=state_dict["states"],
                sampling_params=self.train_sampling_params
                if train else self.test_sampling_params,
                return_logprob=True
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
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                break
            if env_response["next_state"].startswith(state_text + action_text):
                state_dict_delta = self.initialize_state_dict(
                    env_response["next_state"][len(state_text + action_text):]
                )
                for k, v in state_dict_delta.items():
                    state_dict[k].extend(v)
            else:
                tensor_dicts.append(self.get_tensor_dict(state_dict))
                state_dict = self.initialize_state_dict(env_response["next_state"])
            state_text = env_response["next_state"]

        metric["n_turns"].append(turn)
        metric["scores"].append(sum(scores))
        last_response = self.tokenizer.decode(state_dict["states"])

        return tensor_dicts, metric, last_response

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):

        # The data is distributed from rank 0 before each worker operation
        # and gathered before the next operation, which facilitates to do
        # model-agnostic operations, e.g., computing advantages, globally
        # and guarantees the load balancing across all model computations.
        if self.device_mesh["tp"].get_local_rank() == 0:

            data_list = split_and_scatter_list(
                data_list, process_group=self.device_mesh["dp"]
            )
            loop = asyncio.get_event_loop()
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

            all_tensor_dicts, metrics, _ = map(list, zip(*outputs))

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
    def update(self, named_tensor_generator):

        if self.lora_enabled:
            # LoRA update path
            if self.device_mesh["tp"].get_local_rank() == 0:
                # Convert generator to dict (LoRA state is small, safe to load all)
                state_dict = dict(named_tensor_generator)

                # Save LoRA adapters to disk
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    lora_save_path = os.path.join(temp_dir, "lora_adapters")
                    os.makedirs(lora_save_path, exist_ok=True)

                    if dist.get_rank() == 0:
                        print(f"[LoRA Update] Saving LoRA state dict to {lora_save_path}")

                    # Save state_dict (LoRA weights only from PEFT model)
                    torch.save(state_dict, os.path.join(lora_save_path, "adapter_model.bin"))

                    # Delete state_dict to free memory
                    del state_dict
                    torch.cuda.empty_cache()

                    # Unload old LoRA if loaded
                    if self.lora_loaded:
                        if dist.get_rank() == 0:
                            print("[LoRA Update] Unloading previous LoRA adapter...")
                        self.llm.unload_lora_adapter(lora_name="default")
                        self.lora_loaded = False

                    # Load new LoRA
                    if dist.get_rank() == 0:
                        print(f"[LoRA Update] Loading new LoRA adapter from {lora_save_path}...")
                    self.llm.load_lora_adapter(lora_name="default", lora_path=lora_save_path)
                    self.lora_loaded = True

                    if dist.get_rank() == 0:
                        print("[LoRA Update] ✓ LoRA adapter loaded successfully")

                    self.llm.flush_cache()

            dist.barrier()
            return

        # Regular weight update path (non-LoRA)
        dist.barrier()
        # Resume memory BEFORE loading tensors to GPU
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()

        # Don't convert to list - process one tensor at a time to save memory
        tensor_count = 0
        for name, tensor in named_tensor_generator:
            # Keep tensor on CPU, only serialize
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

            # Delete to free memory immediately
            del tensor
            del serialized_tensor

            tensor_count += 1

            if self.device_mesh["tp"].get_local_rank() == 0:
                # Engine will handle GPU loading internally
                self.llm.update_weights_from_tensor(
                    named_tensors=[(
                        name, LocalSerializedTensor(values=serialized_tensors)
                    )],
                    flush_cache=False  # Don't flush until the end
                )
                del serialized_tensors

        # Flush cache after all updates
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.flush_cache()

        torch.cuda.empty_cache()
        dist.barrier()
