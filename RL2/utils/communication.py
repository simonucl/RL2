import os
from datetime import timedelta
import torch
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def broadcast_object(obj, src=None, group=None, group_src=None):
    
    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=group,
        group_src=group_src
    )
    return object_list[0]

def gather_and_concat_list(lst, process_group):

    lists = (
        dist.get_world_size(process_group) * [None]
        if dist.get_rank(process_group) == 0
        else None
    )
    dist.gather_object(
        lst,
        lists,
        group=process_group,
        group_dst=0
    )
    return sum(lists, []) if dist.get_rank(process_group) == 0 else None