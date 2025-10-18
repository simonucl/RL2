"""
LoRA utilities for SGLang server integration.

This module provides functions to manage LoRA adapters with SGLang servers,
including saving adapters from model state, loading them to the server,
and unloading them.
"""

import os
import torch
from typing import Optional, Dict, Callable


def save_lora_adapters(model, save_dir: str, rank: int = 0) -> Optional[str]:
    """
    Save LoRA adapters from a PEFT model to a directory.

    Args:
        model: The PEFT model with LoRA adapters
        save_dir: Directory to save the LoRA adapters
        rank: Process rank (only rank 0 saves)

    Returns:
        Path to saved LoRA adapters if rank==0, None otherwise
    """
    if rank != 0:
        return None

    # Check if model has PEFT config (LoRA enabled)
    if not hasattr(model, 'peft_config'):
        return None

    lora_save_path = os.path.join(save_dir, "lora_adapters")
    os.makedirs(lora_save_path, exist_ok=True)

    # Save LoRA adapters using PEFT's save_pretrained
    model.save_pretrained(lora_save_path)

    return lora_save_path


def load_lora_to_sglang(
    make_request: Callable,
    lora_name: str,
    lora_path: str
) -> None:
    """
    Load LoRA adapter to SGLang server via HTTP API.

    Args:
        make_request: Function to make HTTP requests (from Rollout.make_request)
        lora_name: Name to assign to this LoRA adapter
        lora_path: Path to the LoRA adapter directory
    """
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path
    }
    make_request("load_lora_adapter", payload=payload)


def unload_lora_from_sglang(
    make_request: Callable,
    lora_name: str
) -> None:
    """
    Unload LoRA adapter from SGLang server via HTTP API.

    Args:
        make_request: Function to make HTTP requests (from Rollout.make_request)
        lora_name: Name of the LoRA adapter to unload
    """
    payload = {
        "lora_name": lora_name
    }
    make_request("unload_lora_adapter", payload=payload)
