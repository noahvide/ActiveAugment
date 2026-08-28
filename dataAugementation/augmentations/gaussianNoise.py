from typing import Any, Tuple, Union
from torchvision.transforms.v2 import functional as F
from dataAugementation.intensity import Intensity, IntensityRange
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation

class GaussianNoise(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(0.0, 0.05),
        Intensity.MEDIUM: IntensityRange(0.05, 0.2),
        Intensity.HIGH: IntensityRange(0.2, 0.5),
    }

    def __init__(self, sigma: Union[float, Tuple[float, float], Intensity], **kwargs):
        super().__init__(value=sigma, **kwargs)

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return F.gaussian_noise(inpt, sigma=params["value"])
