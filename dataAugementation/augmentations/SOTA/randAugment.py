from typing import Any, Dict, Tuple, Union
import torch
from torchvision.transforms import v2
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.intensity import Intensity, IntensityRange

class RandAugment(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(0, 10),
        Intensity.MEDIUM: IntensityRange(11, 21),
        Intensity.HIGH: IntensityRange(22, 31)
    }

    def __init__(self, magnitude: Union[float, Tuple[float, float], Intensity], **kwargs):
        """
        Dynamically applies RandAugment with a sampled magnitude.
        """
        super().__init__(value=magnitude, **kwargs)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Applies RandAugment using the sampled magnitude.
        """
        current_magnitude = int(params["value"])
        
        policy = v2.RandAugment(num_ops=2, magnitude=current_magnitude)
        
        is_float = inpt.is_floating_point()
        
        if is_float:
            inpt = (inpt * 255).to(torch.uint8)
            
        inpt = policy(inpt)
        
        if is_float:
            inpt = inpt.to(torch.float32) / 255.0
            
        return inpt