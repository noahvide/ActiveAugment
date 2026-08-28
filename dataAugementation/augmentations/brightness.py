from typing import Any, Dict, Tuple, Union
from torchvision.transforms.v2 import functional as F
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation 
from dataAugementation.intensity import Intensity, IntensityRange

class Brightness(BaseAugmentation):
    CONFIG = {
        Intensity.LOW: IntensityRange(0.5, 0.8),
        Intensity.MEDIUM: IntensityRange(0.8, 1.2),
        Intensity.HIGH: IntensityRange(1.2, 2)
    }

    def __init__(self, brightness: Union[float, Tuple[float, float], Intensity], **kwargs):
        """
        Adjust the brightness of the image.
        
        Args:
            brightness: 0 gives a black image, 1 gives the original image, 
                        and 2 increases the brightness by a factor of 2.
        """
        super().__init__(value=brightness, **kwargs)
    

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        """
        Applies the sampled brightness factor.
        params["value"] is sampled in BaseIntensityTransform.make_params
        """
        
        return F.adjust_brightness(inpt, brightness_factor=params["value"])
    
    


    # def __init__(self, brightness: Union[float, Tuple[float, float], Intensity]):
    #     """
    #     Adjust the brightness of the image.
        
    #     Args:
    #         brightness: 0 gives a black image, 1 gives the original image, 
    #                     and 2 increases the brightness by a factor of 2.
    #     """
    #     super().__init__(value=brightness)