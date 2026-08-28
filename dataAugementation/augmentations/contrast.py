from typing import Any, Dict, Tuple, Union
from torchvision.transforms.v2 import functional as F
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation 
from dataAugementation.intensity import Intensity, IntensityRange

class Contrast(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(0.5, 0.8),
        Intensity.MEDIUM: IntensityRange(0.8, 1.2),
        Intensity.HIGH: IntensityRange(1.2, 3)
    }

    def __init__(self, contrast: Union[float, Tuple[float, float], Intensity], **kwargs):
        """
        Adjust the contrast of the image.
        
        Args:
            contrast: 0 gives a solid gray image, 1 gives the original image, 
                      and values > 1 increase the contrast.
        """
        super().__init__(value=contrast, **kwargs)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Applies the sampled contrast factor.
        params["value"] is sampled per-image in make_params.
        """
        return F.adjust_contrast(inpt, contrast_factor=params["value"])