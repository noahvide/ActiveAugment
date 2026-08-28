from typing import List, Optional, Tuple

import torch
from dataProcessing.dataPath import DataPath
from dataProcessing.datasets.dataset import ClassificationDataset
from models.modelEnums import WeightsStatus

class TrainingDataset(ClassificationDataset):
    def __init__(self, root_dir: DataPath, 
                 weights_status: WeightsStatus,
                 img_size: Tuple[int, int] = (256, 256), 
                 indices: Optional[List[int]] = None):
        
        super().__init__(root_dir, "Train", weights_status, img_size)        

        if indices is not None:
            self.indices = indices
            
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        base_len = len(self.indices)
        original_list_idx = idx % base_len        
        actual_img_idx = self.indices[original_list_idx]
        image, label = self.full_dataset[actual_img_idx]
        image = self.base_transform(image)
        return image, label
