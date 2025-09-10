import torch
from RL2.datasets import BaseDataset, pack_tensor_dicts

class OfflineRLDataset(BaseDataset):
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        tensor_dict = self.tokenize_messages(ex["messages"])
        tensor_dict["label"] = float(ex["label"])
        return tensor_dict
        
    def collate_fn(self, tensor_dicts):
        labels = [tensor_dict.pop("label") for tensor_dict in tensor_dicts]
        packed_dict = pack_tensor_dicts(tensor_dicts)
        packed_dict["labels"] = torch.tensor(labels)
        return packed_dict