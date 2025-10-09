from .fsdp import FSDPActor, FSDPCritic
from .megatron import MegatronActor, MegatronCritic
from .rollout import Rollout

def initialize_actor(config, train):

    if config.backend == "fsdp":
        return FSDPActor(config, train)
    elif config.backend == "megatron":
        return MegatronActor(config, train)
    else:
        raise NotImplementedError

def initialize_critic(config):

    if config.backend == "fsdp":
        return FSDPCritic(config)
    elif config.backend == "megatron":
        return MegatronCritic(config)
    else:
        raise NotImplementedError