import copy
from RL2.datasets.base import BaseDataset


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

        extra_info = ex.get("extra_info", {})
        extra_info["idx"] = idx
        data["extra_info"] = extra_info

        return data

    def collate_fn(self, batch):
        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.config.responses_per_prompt)
        ]