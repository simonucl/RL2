from hydra.core.hydra_config import HydraConfig
from .base import Worker
from .fsdp import FSDPWorker, FSDPActor, FSDPCritic, init_weight_context
from .megatron import MegatronWorker, MegatronActor, MegatronCritic
from .rollout import Rollout

def initialize_actor(config, train):

    from hydra.core.hydra_config import HydraConfig
    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get(
        "actor" if train else "ref_actor"
    )
    if backend == "fsdp":
        from .fsdp import FSDPActor
        return FSDPActor(config, train)
    elif backend == "megatron":
        from .megatron import MegatronActor
        return MegatronActor(config, train)
    else:
        raise NotImplementedError

def initialize_critic(config):

    from hydra.core.hydra_config import HydraConfig
    hydra_config = HydraConfig.get()
    backend = hydra_config.runtime.choices.get("critic")
    if backend == "fsdp":
        from .fsdp import FSDPCritic
        return FSDPCritic(config)
    elif backend == "megatron":
        from .megatron import MegatronCritic
        return MegatronCritic(config)
    else:
        raise NotImplementedError