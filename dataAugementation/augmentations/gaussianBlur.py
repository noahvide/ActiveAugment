from typing import Any, Dict, Tuple, Union
import math
from torchvision.transforms.v2 import functional as F
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation 
from dataAugementation.intensity import Intensity, IntensityRange

class GaussianBlur(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(0, 0.75),
        Intensity.MEDIUM: IntensityRange(0.75, 1.25),
        Intensity.HIGH: IntensityRange(1.25, 2.5)
    }

    def __init__(self, sigma: Union[float, Tuple[float, float], Intensity], **kwargs):
        super().__init__(value=sigma, **kwargs)
        max_sigma = max(r[1] for r in self.resolved_ranges)
         
        k_size = int(math.ceil(max_sigma * 6))
        if k_size % 2 == 0:
            k_size += 1
        
        self.kernel_size = k_size


    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Applies Gaussian Blur.
        params["value"] is the sigma sampled per-image in make_params.
        """
        sigma_val = max(0.001, params["value"])
        
        return F.gaussian_blur(
            inpt, 
            kernel_size=[self.kernel_size, self.kernel_size], 
            sigma=[sigma_val, sigma_val]
        )