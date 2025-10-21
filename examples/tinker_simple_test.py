"""
Simple test script for Tinker PPO trainer.

This creates a minimal dataset to verify the training loop works.
"""

import json
import tempfile
import os

# Create a simple test dataset
test_data = [
    {"prompt": "What is 2+2?", "reward": 1.0},
    {"prompt": "What is the capital of France?", "reward": 0.5},
    {"prompt": "Explain machine learning.", "reward": 0.8},
    {"prompt": "Write a hello world program.", "reward": 1.0},
] * 32  # 128 samples total

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')
    temp_path = f.name

print(f"Created test dataset: {temp_path}")
print(f"Total samples: {len(test_data)}")

print("\nTo run training:")
print(f"python -m RL2.trainer.tinker_ppo \\")
print(f"  train_data.path={temp_path} \\")
print(f"  train_data.prompts_per_rollout=32 \\")
print(f"  tinker.model_name=Qwen/Qwen3-1.7B-Base \\")
print(f"  tinker.max_tokens=128 \\")
print(f"  trainer.n_epochs=1 \\")
print(f"  trainer.save_freq=10 \\")
print(f"  trainer.use_wandb=false")

# Keep the file for manual testing
print(f"\nDataset file: {temp_path}")
print("(Delete manually after testing)")
