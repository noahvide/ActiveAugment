from typing import Any, Dict, Optional, Union
from torchvision.transforms.v2 import functional as F
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.intensity import Intensity

class Flip(BaseAugmentation):
    CONFIG = {}
    
    def __init__(self, orientation: Optional[Union[int, Intensity]] = None, **kwargs):
        """
        Flip the image horizontally (0) or vertically (1). If intensity is passed it is ignored
        """
        if isinstance(orientation, int):
            if orientation == 0 or orientation == 1:
                super().__init__(value=orientation, **kwargs)
            else:
                raise ValueError(f"orientation must be either 0 or 1. Got {orientation}")
        else:    
            super().__init__(value=(0, 1), **kwargs)
    

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["value"] <= 0.5:
            return F.horizontal_flip(inpt=inpt)
        else:
            return F.vertical_flip(inpt=inpt)
            
        
    
    
