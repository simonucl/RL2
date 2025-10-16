import logging
from math_verify import parse, verify

logging.getLogger("math_verify.parser").disabled = True
logging.getLogger("math_verify.grader").disabled = True

async def step(state, action, extra_info):
    try:
        reward = float(
            verify(
                parse(extra_info["answer"]),
                parse(action)
            )
        )
    except Exception as e:
        print(e)
        reward = 0
    return {
        "next_state": None,
        "reward": reward,
        "score": reward,
        "done": True,
        "extra_info": extra_info
    }