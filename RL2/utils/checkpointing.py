import glob
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict
)
from transformers import AutoModelForSequenceClassification
from RL2.utils.offloading import model_offloading_manager

def extract_base_layer_weights(state_dict):
    clean_dict = {}
    
    for name, tensor in state_dict.items():
        if '.base_layer.weight' in name:
            clean_name = name.replace('.base_layer.weight', '.weight')
            clean_dict[clean_name] = tensor
        elif '.weight' in name and 'lora_' not in name and 'base_layer' not in name:
            clean_dict[name] = tensor
            
    return clean_dict

@model_offloading_manager
def get_state_dict(worker, full_state_dict=False):
    if hasattr(worker.model, 'peft_config'):
        with worker.model.summon_full_params(worker.model):
            worker.model.merge_adapter()
            base_model = worker.model.get_base_model()
            options = StateDictOptions(full_state_dict=full_state_dict, cpu_offload=True)
            state_dict = get_model_state_dict(base_model, options=options)
            worker.model.unmerge_adapter()

        state_dict = extract_base_layer_weights(state_dict)
        worker.model.unmerge_adapter()
        return state_dict

    options = StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True
    )
    return get_model_state_dict(worker.model, options=options)

def get_worker_ckpt(worker):

    return {
        "model": get_state_dict(worker),
        "optimizer": worker.optimizer.state_dict(),
        "scheduler": worker.scheduler.state_dict()
    }

def get_ckpt(trainer, workers, step):

    ckpt = {
        "step": step,
        "dataloader": trainer.train_dataloader.state_dict()
    }

    for idx, worker in enumerate(workers):
        if worker.__class__.__name__ in ["Actor", "Critic"]:
            ckpt[f"worker{idx}"] = get_worker_ckpt(worker)

    return ckpt

@model_offloading_manager
def load_worker_ckpt(worker, ckpt):

    if hasattr(worker.model, 'peft_config'):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(
            worker.model, ckpt["model"]
        )
    else:
        set_model_state_dict(
            worker.model, ckpt["model"]
        )
    worker.optimizer.load_state_dict(ckpt["optimizer"])
    worker.scheduler.load_state_dict(ckpt["scheduler"])

def load_ckpt(trainer, workers):

    checkpoint_id = trainer.config.trainer.load_ckpt_from
    if checkpoint_id is None:
        return 0
    if checkpoint_id == "latest":
        save_dirs = glob.glob(f"{trainer.config.trainer.save_dir}/step*")
        if not save_dirs:
            return 0
        checkpoint_id = max(
            save_dirs, key=lambda dir: int(dir.split("/step")[-1])
        )
    
    ckpt = get_ckpt(trainer, workers, 0)
    dcp.load(ckpt, checkpoint_id=checkpoint_id)
    trainer.train_dataloader.load_state_dict(ckpt["dataloader"])
    for idx, worker in enumerate(workers):
        if worker.__class__.__name__ in ["Actor", "Critic"]:
            load_worker_ckpt(worker, ckpt[f"worker{idx}"])
        elif worker.__class__.__name__ == "Rollout":
            if worker.device_mesh["tp"].get_local_rank() == 0:
                worker.llm.release_memory_occupation()
            worker.update(workers[0], ckpt["step"])

    return ckpt["step"]

def save_ckpt(trainer, workers, step):

    if trainer.config.trainer.save_freq is None or step % trainer.config.trainer.save_freq != 0:
        return

    dcp.save(
        get_ckpt(trainer, workers, step),
        checkpoint_id=f"{trainer.config.trainer.save_dir}/step{step}"
    )

def save_model(trainer, worker, rm=False):

    save_dir = trainer.config.trainer.save_dir
    if trainer.config.trainer.save_freq is not None:
        save_dir += "/latest"

    state_dict = get_state_dict(
        worker, full_state_dict=True
    )
    if dist.get_rank() == 0:
        worker.tokenizer.save_pretrained(save_dir)
        # unwrap the model
        model_to_save = worker.model.module
        if rm:
            # For RM, we load token classification model for simplicity 
            # but save sequence classification model for compatibility.
            with torch.device("meta"):
                model_to_save = AutoModelForSequenceClassification.from_config(
                    model_to_save.config
                )
        elif hasattr(worker.model, 'peft_config'):
            model_to_save.save_pretrained(
                save_dir + "/lora_adapters"
            )
            model_to_save = model_to_save.get_base_model()

        model_to_save.save_pretrained(
            save_dir, state_dict=state_dict
        )

    dist.barrier()