import asyncio
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns

NUM_ENVS = 16
ENV_ID = "rg:letter_counting"
WRAPPERS = ""
PROMPT_TEMPLATE = "qwen3_general"
ENV_POOL = []
ENV_LOCKS = []

def apply_no_template(observation):
    return observation

def apply_qwen3_general_template(observation):
    return (
        f"<|im_start|>user\nQuestion: {observation}\nPlease reason step by step,"
        " and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>"
        "assistant\n"
    )

def apply_qwen3_game_template(observation):
    return (
        "<|im_start|>user\nYou are playing language games. Make valid actions to win."
        f"\nObservation: {observation}\nPlease reason step by step, and put your final"
        " answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_code_template(observation):
    return (
        "You are an expert Python programmer. You will be given a question (problem"
        " specification) and will generate a correct Python program that matches the"
        f" specification and passes all tests.\nQuestion: {observation}"
        "\nPlease reason step by step, and write your code in markdown format, e.g.,"
        " ```python\n# YOUR CODE HERE\n```."
    )

TEMPLATE_FACTORY = {
    "no": apply_no_template,
    "qwen3_general": apply_qwen3_general_template,
    "qwen3_game": apply_qwen3_game_template,
    "code": apply_code_template,
}
    
for idx in range(NUM_ENVS):
    env = gem.make(env_id=ENV_ID, seed=233 + idx)
    wrappers = get_wrapper_fns(WRAPPERS, tokenizer=None)
    for wrapper in wrappers:
        env = wrapper(env)
    ENV_POOL.append(env)
    ENV_LOCKS.append(asyncio.Lock())

async def reset(extra_info):

    env_idx = extra_info["idx"] % NUM_ENVS
    await ENV_LOCKS[env_idx].acquire()
    state, _ = ENV_POOL[env_idx].reset()
    return TEMPLATE_FACTORY[PROMPT_TEMPLATE](state)

async def step(state, action, extra_info):

    env_idx = extra_info["idx"] % NUM_ENVS

    (
        next_state,
        reward,
        terminated,
        truncated,
        _
    ) = ENV_POOL[env_idx].step(action)
    next_state = TEMPLATE_FACTORY[PROMPT_TEMPLATE](next_state)
    done = terminated or truncated
    
    if done:
        ENV_LOCKS[env_idx].release()

    return {
        "next_state": next_state,
        "reward": reward,
        "score": reward,
        "done": done,
        "extra_info": extra_info
    }