from torch.utils.data import Dataset, Subset
from typing import Any, Dict

class CachedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        
        underlying_dataset = base_dataset
        while hasattr(underlying_dataset, 'dataset') and isinstance(underlying_dataset, Subset):
            underlying_dataset = underlying_dataset.dataset
            
        self.classes = underlying_dataset.classes # type: ignore
        
        self.norm_transform = underlying_dataset.norm_transform # type: ignore
        
        self.cache: Dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.base_dataset) # type: ignore

    def __getitem__(self, idx: int) -> Any:
        if idx in self.cache:
            return self.cache[idx]
        
        data = self.base_dataset[idx]
        
        self.cache[idx] = data
        
        return data