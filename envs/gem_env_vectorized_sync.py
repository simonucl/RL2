import random
import time
import logging
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns
import asyncio
from collections import deque

NUM_ENVS = 16
GAME = "game:GuessTheNumber-v0"
WRAPPERS = "concat"
# GAME = "rg:letter_counting"
# WRAPPERS = ""
PROMPT_TEMPLATE = "qwen3_general"

def apply_qwen3_game_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing language games. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_no_template(observation: str) -> str:
    return observation

def apply_qwen3_general_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nQuestion: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def apply_code_template(observation: str) -> str:
    return (
        "You are an expert Python programmer. "
        "You will be given a question (problem specification) and will generate a correct "
        "Python program that matches the specification and passes all tests."
        f"\nQuestion: {observation}"
        "\nPlease reason step by step, and write your code in markdown format, e.g., ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "qwen3_game": apply_qwen3_game_template,
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "code": apply_code_template,
}

class VectorizedGemEnv:
    
    def __init__(self):
        self.num_envs = NUM_ENVS
        print(f"Initializing {NUM_ENVS} vectorized GEM environments...")
        
        # --- NEW: More robust state management ---
        self.available_buckets = asyncio.Queue()
        for i in range(self.num_envs):
            self.available_buckets.put_nowait(i)
        
        # Mapping and lock for episode-to-bucket assignment
        self.episode_to_bucket = {}
        self.episode_to_bucket_lock = asyncio.Lock()

        # --- REFACTORED: Smarter Cohort Management ---
        self.cohort_id_counter = 0
        self.episode_to_cohort_id = {}
        self.cohorts = {}  # cohort_id -> {
                           #   "episodes": set(), "done_episodes": set(),
                           #   "pending_actions": {}, "pending_futures": {}, "lock": asyncio.Lock()
                           # }
        self.cohort_lock = asyncio.Lock() # Global lock for creating/deleting cohorts

        # --- REMOVED: Old batch counters and global locks ---

        # State for managing reset batching
        self.reset_pending_futures = {} # bucket_idx -> future
        self.reset_pending_episodes = {} # bucket_idx -> episode_idx
        self.reset_lock = asyncio.Lock()

        self.vec_env = gem.make_vec(
            [GAME] * NUM_ENVS,
            vec_kwargs=[{"seed": 233 + i} for i in range(NUM_ENVS)],
            wrappers=get_wrapper_fns(WRAPPERS if WRAPPERS else "", tokenizer=None),
            async_mode=True,
        )
        print(f"Successfully initialized {NUM_ENVS} vectorized GEM environments.")

    async def reset_episode(self, extra_info: Dict[str, Any]) -> str:
        episode_idx = extra_info['idx']
        
        logging.info(f"[Reset] Episode {episode_idx} waiting for an available bucket...")
        bucket_idx = await self.available_buckets.get()
        logging.info(f"[Reset] Episode {episode_idx} acquired bucket {bucket_idx}.")
        
        async with self.episode_to_bucket_lock:
            self.episode_to_bucket[episode_idx] = bucket_idx

        future = asyncio.get_running_loop().create_future()
        
        async with self.reset_lock:
            self.reset_pending_futures[bucket_idx] = future
            self.reset_pending_episodes[bucket_idx] = episode_idx
            
            # If the reset batch is full, trigger it
            if len(self.reset_pending_episodes) == self.num_envs:
                logging.info("[Reset] Batch is full. Triggering vec_env.reset() and creating new cohort.")
                
                # Create the new cohort
                async with self.cohort_lock:
                    cohort_id = self.cohort_id_counter
                    self.cohort_id_counter += 1
                    
                    episodes_in_cohort = set(self.reset_pending_episodes.values())
                    self.cohorts[cohort_id] = {
                        "episodes": episodes_in_cohort,
                        "done_episodes": set(),
                        "pending_actions": {},
                        "pending_futures": {},
                        "lock": asyncio.Lock()
                    }
                    for ep_idx in episodes_in_cohort:
                        self.episode_to_cohort_id[ep_idx] = cohort_id
                    logging.info(f"[Reset] Created Cohort {cohort_id} with episodes: {episodes_in_cohort}")

                observations, _ = self.vec_env.reset()
                
                # Distribute results
                # for b_idx, obs in enumerate(observations):
                for bucket_idx in range(self.num_envs):
                    # Get the observation for this specific bucket
                    obs = observations[bucket_idx]
                    
                    # Find which episode and future are waiting for this bucket's result
                    ep_idx = self.reset_pending_episodes[bucket_idx]
                    fut = self.reset_pending_futures[bucket_idx]
                    
                    template_fn = TEMPLATE_FACTORY.get(PROMPT_TEMPLATE, apply_no_template)
                    formatted_obs = template_fn(obs)
                    
                    # DO NOT change the episode_to_bucket mapping here. It's already correct.
                    # self.episode_to_bucket[ep_idx] = b_idx # This was the bug!
                    
                    if not fut.done():
                        fut.set_result(formatted_obs)
                
                self.reset_pending_futures.clear()
                self.reset_pending_episodes.clear()

        return await future

    async def step_episode(self, state: str, action: str, extra_info: Dict[str, Any]) -> Dict[str, Any]:
        episode_idx = extra_info['idx']

        # Find the episode's cohort
        if episode_idx not in self.episode_to_cohort_id:
            raise RuntimeError(f"Episode {episode_idx} has no cohort. Did you reset it properly?")
        cohort_id = self.episode_to_cohort_id[episode_idx]
        
        async with self.cohort_lock:
            if cohort_id not in self.cohorts:
                # This can happen if the cohort was just cleared. Return a dummy done response.
                return {"next_state": "", "reward": 0.0, "score": 0.0, "done": True, "extra_info": {}}
            cohort = self.cohorts[cohort_id]

        future = asyncio.get_running_loop().create_future()
        
        async with cohort["lock"]:
            # Store the action and future for this active episode
            cohort["pending_actions"][episode_idx] = action
            cohort["pending_futures"][episode_idx] = future
            
            # Check if all ACTIVE episodes have submitted their actions
            active_episodes = cohort["episodes"] - cohort["done_episodes"]
            if set(cohort["pending_actions"].keys()) == active_episodes:
                logging.info(f"[Step] All active episodes in Cohort {cohort_id} have submitted actions. Triggering step.")
                
                # --- Batch is ready, let's go! ---
                step_actions = [""] * self.num_envs  # Default to no-op
                
                # Fill in actions for all cohort members
                async with self.episode_to_bucket_lock:
                    for ep_idx in cohort["episodes"]:
                        bucket_idx = self.episode_to_bucket[ep_idx]
                        if ep_idx in active_episodes:
                            step_actions[bucket_idx] = cohort["pending_actions"][ep_idx]
                
                # Clear pending actions for the next turn
                pending_futures_copy = dict(cohort["pending_futures"])
                cohort["pending_actions"].clear()
                cohort["pending_futures"].clear()

                # Perform the actual step
                obs, rewards, terminateds, truncateds, infos = self.vec_env.step(step_actions)

                # Process results and check for newly done episodes
                newly_done_episodes = set()
                async with self.episode_to_bucket_lock:
                    for ep_idx in active_episodes:
                        bucket_idx = self.episode_to_bucket[ep_idx]
                        is_done = terminateds[bucket_idx] or truncateds[bucket_idx]
                        
                        template_fn = TEMPLATE_FACTORY.get(PROMPT_TEMPLATE, apply_no_template)
                        result_dict = {
                            "next_state": template_fn(obs[bucket_idx]),
                            "reward": rewards[bucket_idx],
                            "score": rewards[bucket_idx],
                            "done": is_done,
                            "extra_info": {**infos[bucket_idx], "idx": ep_idx},
                        }
                        
                        fut = pending_futures_copy[ep_idx]
                        if not fut.done():
                            fut.set_result(result_dict)
                        
                        if is_done:
                            newly_done_episodes.add(ep_idx)
                
                cohort["done_episodes"].update(newly_done_episodes)

                # Check if the entire cohort is now finished
                if cohort["done_episodes"] == cohort["episodes"]:
                    logging.info(f"[Step] Cohort {cohort_id} is fully done. Releasing all buckets.")
                    async with self.cohort_lock, self.episode_to_bucket_lock:
                        for ep_idx in cohort["episodes"]:
                            bucket_to_release = self.episode_to_bucket.pop(ep_idx, None)
                            if bucket_to_release is not None:
                                await self.available_buckets.put(bucket_to_release)
                            self.episode_to_cohort_id.pop(ep_idx, None)
                        self.cohorts.pop(cohort_id, None)

        return await future

VECTORIZED_ENV = VectorizedGemEnv()

async def reset(extra_info: Dict[str, Any]) -> str:
    episode_idx = extra_info['idx']
    logging.info(f"[API] Reset called for episode {episode_idx}")
    try:
        observation = await VECTORIZED_ENV.reset_episode(extra_info)
        logging.info(f"[API] Reset completed for episode {episode_idx}")
        return observation
    except Exception as e:
        logging.error(f"[API] Error resetting episode {episode_idx}: {e}")
        raise

async def step(state: str, action: str, extra_info: Dict[str, Any]) -> Dict[str, Any]:    
    episode_idx = extra_info['idx']
    # print(f"[API] Step called for episode {episode_idx}")
    logging.info(f"[API] Step called for episode {episode_idx}")
    try:
        result = await VECTORIZED_ENV.step_episode(state, action, extra_info)
        logging.info(f"[API] Step completed for episode {episode_idx}")
        return result
    except Exception as e:
        logging.error(f"[API] Error stepping episode {episode_idx}: {e}")
        raise