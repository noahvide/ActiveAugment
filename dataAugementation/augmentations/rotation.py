from typing import Any, Dict, Tuple, Union
from torchvision.transforms.v2 import functional as F
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.intensity import Intensity, IntensityRange

class Rotation(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(-15, 15),
        Intensity.MEDIUM: (IntensityRange(-30, -15), IntensityRange(15, 30)),
        Intensity.HIGH: (IntensityRange(-45, -30), IntensityRange(30, 45))
    }
    
    def __init__(self, angle: Union[float, Tuple[float, float], Intensity], **kwargs):
        super().__init__(value=angle, **kwargs)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.rotate(inpt, angle=params["value"])