import os
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))

    local_rank = int(os.environ["LOCAL_RANK"])
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)

def gather_and_concat_list(lst, device_mesh):

    lists = (
        device_mesh.size() * [None]
        if device_mesh.get_local_rank() == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=device_mesh.get_group(),
        group_dst=0
    )
    return sum(lists, []) if device_mesh.get_local_rank() == 0 else None