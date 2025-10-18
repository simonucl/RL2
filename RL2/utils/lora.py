"""
LoRA utilities for SGLang server integration.

This module provides functions to manage LoRA adapters with SGLang servers,
including saving adapters from model state, loading them to the server,
and unloading them.
"""

from typing import Callable

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
