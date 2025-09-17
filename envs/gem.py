import asyncio
import gem
from gem.wrappers.wrapper_factory import get_wrapper_fns

NUM_ENVS = 4
ENV_ID = "game:Sudoku-v0-random"
WRAPPERS = "concat"
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
    
vec_env = gem.make_vec(
    [ENV_ID] * NUM_ENVS,
    vec_kwargs=[{"seed": 233 + idx} for idx in range(NUM_ENVS)],
    wrappers=get_wrapper_fns(WRAPPERS, tokenizer=None),
    async_mode=True
)

async def reset(extra_info):
    """Reset all environments and return initial observations with templates applied."""
    states, _ = vec_env.reset()
    return [TEMPLATE_FACTORY[PROMPT_TEMPLATE](state) for state in states]

async def step(states, actions, extra_info):
    """Step all environments with given actions and return batch results."""
    next_states, rewards, terminated, truncated, _ = vec_env.step(actions)
    # Apply templates to next states for consistency
    templated_next_states = [TEMPLATE_FACTORY[PROMPT_TEMPLATE](state) for state in next_states]
    done = terminated | truncated
    return {
        "next_state": templated_next_states,
        "reward": rewards,
        "done": done,
        "extra_info": extra_info,
    }