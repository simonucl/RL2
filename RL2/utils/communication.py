import os
from datetime import timedelta
import torch
import torch.distributed as dist
import math

def initialize_global_process_group(timeout_second=36000):
    
    dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def _unwrap_process_group(process_group):
    if hasattr(process_group, '_wrapped'):
        process_group = process_group._wrapped
    elif hasattr(process_group, 'group'):
        process_group = process_group.group
    return process_group

def split_and_scatter_list(lst, process_group):

    if process_group.get_local_rank() == 0:
        length_per_dp = math.ceil(len(lst) / process_group.size())
    lists = [
        lst[rank * length_per_dp:(rank + 1) * length_per_dp]
        if process_group.get_local_rank() == 0 else None
        for rank in range(process_group.size())
    ]
    lst = [None]
    dist.scatter_object_list(
        lst,
        lists,
        group=process_group.get_group(),
        group_src=0
    )
    return lst[0]

def broadcast_object(obj, src=None, group=None, group_src=None):
    object_list = [obj]
    dist.broadcast_object_list(
        object_list,
        src=src,
        group=_unwrap_process_group(group),
        group_src=group_src
    )
    return object_list[0]

def gather_and_concat_list(lst, process_group):

    # process_group = _unwrap_process_group(process_group)
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