import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp._runtime_utils import _lazy_init
# TODO: why offloading is incompatible with initialization on meta device?
def init_weight_context(worker):
    if any([
        dist.get_rank() == 0,
        worker.device_mesh["tp"].size() > 1 and worker.device_mesh["tp"].get_local_rank() == 0,
        getattr(worker.config, "offload_model", False)
    ]):
        return torch.device("cpu")
    return torch.device("meta")

def load_model_to_device(worker, device):
    
    if not getattr(worker.config, "offload_model", False):
        return

    _lazy_init(worker.model, worker.model)
    for handle in worker.model._all_handles:
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(device, non_blocking=True)
        flat_param._local_shard = flat_param.data

def load_optimizer_to_device(worker, device):

    if not getattr(worker.config, "offload_optimizer", False):
        return

    for param_group in worker.optimizer.param_groups:
        for param in param_group["params"]:
            state = worker.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(
                        device, non_blocking=True
                    )

def model_offloading_manager(func):

    @functools.wraps(func)
    def func_with_model_offloading(worker, *args, **kwargs):
        load_model_to_device(worker, torch.cuda.current_device())
        output = func(worker, *args, **kwargs)
        load_model_to_device(worker, "cpu")
        return output
    
    return func_with_model_offloading

def optimizer_offloading_manager(func):

    @functools.wraps(func)
    def func_with_optimizer_offloading(worker, *args, **kwargs):
        load_optimizer_to_device(worker, torch.cuda.current_device())
        output = func(worker, *args, **kwargs)
        load_optimizer_to_device(worker, "cpu")
        return output
    
    return func_with_optimizer_offloading