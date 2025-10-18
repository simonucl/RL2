from omegaconf import OmegaConf
import os
import time
import socket
import requests
import multiprocessing
import torch.distributed as dist
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server
from sglang_router.launch_router import RouterArgs, launch_router

try:
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
except: # older version of SGLang
    from sglang.srt.patch_torch import monkey_patch_torch_reductions

def get_host():

    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def get_available_port():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]

def prepare_environment_variables(process_group):

    if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
        del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
    monkey_patch_torch_reductions()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        cuda_visible_devices = cuda_visible_devices.split(",")
        cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
    else:
        cuda_visible_device = os.environ["LOCAL_RANK"]
    cuda_visible_devices = dist.get_world_size(process_group) * [None]
    dist.all_gather_object(
        cuda_visible_devices,
        cuda_visible_device,
        process_group,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

def launch_server_process(server_args, enable_lora=False, max_lora_rank=64):

    server_args = OmegaConf.to_container(server_args)

    # Build server kwargs
    server_kwargs = {
        'enable_memory_saver': True,
        'host': get_host(),
        'port': get_available_port(),
        'log_level': 'error',
        **server_args
    }

    # Add LoRA configuration if enabled
    if enable_lora:
        server_kwargs['enable_lora'] = True
        server_kwargs['max_loras_per_batch'] = 1
        server_kwargs['max_lora_rank'] = max_lora_rank

    server_args = ServerArgs(**server_kwargs)
    process = multiprocessing.Process(
        target=launch_server, args=(server_args,)
    )
    process.start()

    with requests.Session() as session:
        while True:
            assert process.is_alive()
            try:
                response = session.get(
                    f"{server_args.url()}/health_generate"
                )
                if response.status_code == 200:
                    return server_args.url()
            except:
                pass
            time.sleep(1)

def launch_router_process(worker_urls):

    router_args = RouterArgs(
        worker_urls=worker_urls,
        host=get_host(),
        port=get_available_port(),
        log_level="error"
    )
    process = multiprocessing.Process(
        target=launch_router, args=(router_args,)
    )
    process.start()
    time.sleep(3)
    assert process.is_alive()
    return f"http://{router_args.host}:{router_args.port}"