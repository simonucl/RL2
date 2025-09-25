import copy
import torch
from RL2.datasets.base import BaseDataset, get_tensor_dict


class RLDataset(BaseDataset):

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        data = {}

        if "prompt" in ex.keys():
            data["prompt"] = ex["prompt"]
        elif "messages" in ex.keys():
            data["prompt"] = self.tokenizer.apply_chat_template(
                ex["messages"],
                add_generation_prompt=True,
                tokenize=False
            )

        data["extra_info"] = ex.get("extra_info", {})
        return data

    def collate_fn(self, data_list):
        return [
            copy.deepcopy(data)
            for data in data_list
            for _ in range(self.config.responses_per_prompt)
        ]


def initialize_state_dict(tokenizer, state_text):

    states = tokenizer.encode(state_text, add_special_tokens=False)
    return {
        "states": states,
        "actions": len(states) * [0],
        "action_mask": len(states) * [0],
        "logps": len(states) * [0],
        "rewards": len(states) * [0]
    }

def state_dict_to_tensor_dict(state_dict):

    tensor_dict = get_tensor_dict(
        state_dict["states"],
        state_dict["actions"],
        state_dict["action_mask"]
    )
    tensor_dict["llm_logps"] = torch.FloatTensor(state_dict["logps"][1:])
    tensor_dict["rewards"] = torch.FloatTensor(state_dict["rewards"][1:])
    return tensor_dict