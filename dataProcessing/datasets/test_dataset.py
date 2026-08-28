
from typing import Dict, List, Tuple

import torch
from dataAugementation.augmentations import AUGMENTATION_MAP, AugmentationEnum
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.intensity import INTENSITY_MAP, Intensity
from dataProcessing.dataPath import DataPath
from dataProcessing.datasets.dataset import ClassificationDataset
from models.modelEnums import WeightsStatus


class TestDataset(ClassificationDataset):
    def __init__(self, 
                 root_dir: DataPath, 
                 weights_status: WeightsStatus,
                 img_size: Tuple[int, int] = (256, 256)):
        
        super().__init__(root_dir, "Test", weights_status, img_size)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        image, label = self.full_dataset[idx]
        image = self.base_transform(image)
        return image, label

# class TestDataset(ClassificationDataset):
#     def __init__(self, root_dir: DataPath, 
#                  img_size: Tuple[int, int] = (256, 256), 
#                  weights_status: WeightsStatus = WeightsStatus.RANDOM):
        
#         super().__init__(root_dir, "Test", img_size, 
#                          weights_status)
#         self.test_variants: list[BaseAugmentation] = []
#         for name, augmentation in AUGMENTATION_MAP.items():
#             if name == AugmentationEnum.NONE:
#                 continue
#             for _, intensity in INTENSITY_MAP.items():
#                 if intensity != Intensity.ALL:
#                     self.test_variants.append(augmentation(intensity))
        
#         self.num_variants = len(self.test_variants)


#     def __len__(self):
#         base_len = len(self.indices)
#         return base_len * (1 + self.num_variants)
        
#     def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
#         base_len = len(self.indices)
#         variant_idx = idx // base_len         
#         actual_img_idx = self.indices[idx % base_len]
#         image, label = self.full_dataset[actual_img_idx]
        
#         image = self.base_transform(image)
        
#         if variant_idx > 0:
#             augmentation : BaseAugmentation = self.test_variants[variant_idx - 1]
#             image = augmentation(image)
#             image = torch.clamp(image, 0.0, 1.0)
#             image = self.norm_transform(image)
#             return image, label
#         else:
#             image = self.norm_transform(image)
#             return image, label

