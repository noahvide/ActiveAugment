from typing import Any, Dict
import torch
from torchvision.transforms import v2
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation

class TrivialAugment(BaseAugmentation):
    CONFIG = {}
    def __init__(self, val = None, **kwargs):
        super().__init__(0, **kwargs)
        self.policy = v2.TrivialAugmentWide()

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        is_float = inpt.is_floating_point()

        if is_float:
            inpt = (inpt * 255).to(torch.uint8)
            
        img = self.policy(inpt)
        
        if is_float:
            img = img.to(torch.float32) / 255.0
            
        return img