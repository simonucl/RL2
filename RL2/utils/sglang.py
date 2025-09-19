from omegaconf import OmegaConf
import time
import socket
import requests
import multiprocessing
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