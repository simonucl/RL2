from hydra.core.hydra_config import HydraConfig
from .fsdp import FSDPActor, FSDPCritic
from .megatron import MegatronActor, MegatronCritic
from .rollout import Rollout

hydra_config = HydraConfig.get()

def initialize_actor(config, train):

    backend = hydra_config.runtime.choices.get(
        "actor" if train else "ref_actor"
    )
    if backend == "fsdp":
        return FSDPActor(config, train)
    elif backend == "megatron":
        return MegatronActor(config, train)
    else:
        raise NotImplementedError

def initialize_critic(config):

    backend = hydra_config.runtime.choices.get("critic")
    if backend == "fsdp":
        return FSDPCritic(config)
    elif backend == "megatron":
        return MegatronCritic(config)
    else:
        raise NotImplementedError