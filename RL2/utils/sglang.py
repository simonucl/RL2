from omegaconf import OmegaConf
import os
import time
import socket
import asyncio
import aiohttp
import requests
import multiprocessing
import torch.distributed as dist
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import launch_server
from sglang_router.launch_router import RouterArgs, launch_router

def get_host():

    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def get_available_port():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]

def prepare_environment_variables(device_mesh):

    if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
        del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
    monkey_patch_torch_reductions()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices:
        cuda_visible_devices = cuda_visible_devices.split(",")
        cuda_visible_device = cuda_visible_devices[int(os.environ["LOCAL_RANK"])]
    else:
        cuda_visible_device = os.environ["LOCAL_RANK"]
    cuda_visible_devices = device_mesh.size() * [None]
    dist.all_gather_object(
        cuda_visible_devices,
        cuda_visible_device,
        device_mesh.get_group(),
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

def launch_server_process(server_args):

    server_args = OmegaConf.to_container(server_args)
    server_args = ServerArgs(
        enable_memory_saver=True,
        host=get_host(),
        port=get_available_port(),
        **server_args
    )
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
        port=get_available_port()
    )
    process = multiprocessing.Process(
        target=launch_router, args=(router_args,)
    )
    process.start()
    time.sleep(3)
    assert process.is_alive()
    return f"http://{router_args.host}:{router_args.port}"

def make_request(url, endpoint, payload=None):

    response = requests.post(
        f"{url}/{endpoint}",
        json=payload or {}
    )
    response.raise_for_status()

async def async_generate(url, states, sampling_params):

    payload = {
        "input_ids": states,
        "sampling_params": sampling_params,
        "return_logprob": True
    }

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.post(
                    f"{url}/generate",
                    json=payload
                ) as response:
                    return await response.json(content_type=None)
            except:
                await asyncio.sleep(1)