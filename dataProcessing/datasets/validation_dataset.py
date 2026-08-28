
from typing import Tuple

import torch
from dataProcessing.dataPath import DataPath
from dataProcessing.datasets.dataset import ClassificationDataset
from models.modelEnums import WeightsStatus


class ValidationDataset(ClassificationDataset):
    def __init__(self, root_dir: DataPath, 
                weights_status: WeightsStatus,
                img_size: Tuple[int, int] = (256, 256)
                ):
    
        super().__init__(root_dir, "Val", weights_status, img_size)
    
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, label = self.full_dataset[idx]
        image = self.base_transform(image)
        return image, label
        # image = self.norm_transform(image)