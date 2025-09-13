import re

async def step(state, action, extra_info):

    env_response = {
        "next_state": None,
        "reward": 0.0,
        "score": 0.0,
        "done": True,
        "extra_info": extra_info
    }

    match = re.search(r"<answer>(.*?)</answer>", action)
    if match is None:
        return env_response
    equation = match.group(1).strip()
    env_response["reward"] = 0.1
    
    try:
        # maybe the number cannot be converted to integer
        numbers = [int(n) for n in re.findall(r"\d+", equation)]
        assert sorted(numbers) == sorted(extra_info["numbers"])
    except:
        return env_response
        
    try:
        assert re.match(r"^[\d+\-*/().\s]+$", equation)
        # maybe the equation is illegal
        result = eval(equation, {"__builtins__": None}, {})
        assert abs(result - extra_info["target"]) < 1e-5
        env_response["reward"] = 1.0
        env_response["score"] = 1.0
    except:
        pass
    return env_response