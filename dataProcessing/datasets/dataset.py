from abc import abstractmethod
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2._transform import Transform
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from dataProcessing.dataPath import DataPath
from dataProcessing.statsRegistry import get_dataset_stats
from models.modelEnums import WeightsStatus


class ClassificationDataset(Dataset):
    def __init__(self, 
        root_dir: DataPath, 
        split: Literal["Train", "Val", "Test"], 
        weights_status: WeightsStatus,
        img_size: Tuple[int, int] = (256, 256),
    ):
        
        if not isinstance(root_dir, DataPath):
            raise TypeError(f"root_dir must be an instance of DataPath, got {type(root_dir)}")
        
        self.split = split
        self.root = Path(root_dir.get_path) / split
        pretrained = weights_status in ["frozen", "tunable"]

        base_list: List[Transform] = [v2.ToImage()]
        if pretrained:
            base_list.append(v2.RGB())
            
        base_list.extend([
            v2.Resize(img_size, antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        stats = get_dataset_stats(root_dir)
        mean = stats["mean"]
        std = stats["std"]
        self.mean = mean
        self.std = std
        
        self.base_transform = v2.Compose(base_list)
        self.norm_transform = v2.Normalize(mean, std)

        self.full_dataset = datasets.ImageFolder(root=str(self.root))
        self.indices = list(range(len(self.full_dataset)))
    
    @property
    def targets(self) -> List[int]:
        return self.full_dataset.targets
    
    @property
    def classes(self) -> List[str]:
        return self.full_dataset.classes

    def __len__(self):
        return len(self.indices)

    @abstractmethod
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        pass




