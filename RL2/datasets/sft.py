from RL2.datasets import BaseDataset, pack_tensor_dicts

class SFTDataset(BaseDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        if "prompt" in ex.keys():
            tensor_dict = self.tokenize_prompt_response(
                ex["prompt"], ex["response"]
            )
        else:
            tensor_dict = self.tokenize_messages(ex["messages"])
        return tensor_dict
        
    def collate_fn(self, tensor_dicts):
        return pack_tensor_dicts(tensor_dicts)