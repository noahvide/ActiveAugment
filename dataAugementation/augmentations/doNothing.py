from typing import Any, Dict
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation 

class DoNothing(BaseAugmentation):
    CONFIG = {}
    def __init__(self, val = None,  **kwargs):
        super().__init__(value=0, **kwargs)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return inpt
