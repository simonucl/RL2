import torch
from RL2.datasets import BaseDataset, pack_tensor_dicts

class OfflineRLDataset(BaseDataset):
    
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        
        # Filter dataset based on label threshold if specified
        if hasattr(config, 'label_threshold') and config.label_threshold is not None:
            original_size = len(self.dataset)
            filtered_indices = [
                i for i, ex in enumerate(self.dataset) 
                if float(ex["label"]) >= config.label_threshold
            ]
            self.dataset = self.dataset.select(filtered_indices)
            print(f"Filtered dataset from {original_size} to {len(self.dataset)} examples "
                  f"with label_threshold={config.label_threshold}")
    
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