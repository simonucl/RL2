"""
LoRA utilities for SGLang server integration.

This module provides functions to manage LoRA adapters with SGLang servers,
including saving adapters from model state, loading them to the server,
and unloading them.
"""

import os
import requests
import torch
import torch.distributed as dist
from typing import Optional, Dict


def save_lora_adapters(model, save_dir: str, rank: int = 0):
    if rank != 0:
        return None

    if not hasattr(model, 'peft_config'):
        return None

    lora_save_path = os.path.join(save_dir, "lora_adapters")
    os.makedirs(lora_save_path, exist_ok=True)

    model.save_pretrained(lora_save_path)

    return lora_save_path


def load_lora_to_sglang_server(
    worker_url: str,
    lora_name: str,
    lora_path: str,
    max_retries: int = 3
) -> bool:
    payload = {
        "lora_name": lora_name,
        "lora_path": lora_path
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{worker_url}/load_lora_adapter",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                print(f"Failed to load LoRA adapter after {max_retries} attempts: {e}")
                return False
            import time
            time.sleep(1)

    return False


def unload_lora_from_sglang_server(
    worker_url: str,
    lora_name: str,
    max_retries: int = 3
) -> bool:
    payload = {
        "lora_name": lora_name
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{worker_url}/unload_lora_adapter",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                # It's okay if unload fails (e.g., adapter not loaded)
                print(f"Note: Failed to unload LoRA adapter (may not be loaded): {e}")
                return False
            import time
            time.sleep(1)

    return False