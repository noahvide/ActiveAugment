from typing import Any, Dict
import torch
from torchvision.transforms import v2
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataProcessing.dataPath import DataPath

class AutoAugment(BaseAugmentation):
    CONFIG = {}
    def __init__(self, value=None, *, dataset_name: DataPath, **kwargs):
        """
        Dynamically applies AutoAugment.
        """
        super().__init__(0, dataset_name=dataset_name, **kwargs)

        policy_mapping = {
            "CIFAR_10": v2.AutoAugmentPolicy.CIFAR10,
            "CIFAR_100": v2.AutoAugmentPolicy.CIFAR10,
            "MNIST": v2.AutoAugmentPolicy.SVHN,
            "STL": v2.AutoAugmentPolicy.IMAGENET,
            "BRISC": v2.AutoAugmentPolicy.IMAGENET,
            "BUSI": v2.AutoAugmentPolicy.IMAGENET,
            "FETAL_PLANES": v2.AutoAugmentPolicy.IMAGENET,
            "ISIC": v2.AutoAugmentPolicy.IMAGENET
        }
        
        chosen_policy = policy_mapping[dataset_name.value]
        self.policy = v2.AutoAugment(policy=chosen_policy)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_float = inpt.is_floating_point()
        
        if is_float:
            inpt = (inpt * 255).to(torch.uint8)
            
        inpt = self.policy(inpt)
        
        if is_float:
            inpt = inpt.to(torch.float32) / 255.0
            
        return inpt