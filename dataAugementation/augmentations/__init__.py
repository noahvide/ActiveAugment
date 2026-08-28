from enum import Enum
from typing import Dict
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.augmentations.brightness import Brightness
from dataAugementation.augmentations.contrast import Contrast
from dataAugementation.augmentations.gaussianBlur import GaussianBlur
from dataAugementation.augmentations.gaussianNoise import GaussianNoise
from dataAugementation.augmentations.rotation import Rotation
from dataAugementation.augmentations.flip import Flip
from dataAugementation.augmentations.doNothing import DoNothing

from dataAugementation.augmentations.SOTA import TrivialAugment, RandAugment, AutoAugment

class AugmentationEnum(str, Enum):
    ROTATION = "Rotation"
    GAUSSIANBLUR = "GaussianBlur"
    GAUSSIANNOISE = "GaussianNoise"
    CONTRAST = "Contrast"
    BRIGHTNESS = "Brightness"
    FLIP = "Flip"
    TRIVIALAUG = "TrivialAugment"
    RANDAUG = "RandAugment"
    AUTOAUG = "AutoAugment"
    NONE = "None"


AUGMENTATION_MAP : Dict[AugmentationEnum, type[BaseAugmentation]] = {
    AugmentationEnum.ROTATION: Rotation,
    AugmentationEnum.GAUSSIANBLUR: GaussianBlur,
    AugmentationEnum.GAUSSIANNOISE: GaussianNoise,
    AugmentationEnum.CONTRAST: Contrast,
    AugmentationEnum.BRIGHTNESS: Brightness,
    AugmentationEnum.FLIP: Flip,
    AugmentationEnum.TRIVIALAUG: TrivialAugment,
    AugmentationEnum.RANDAUG: RandAugment,
    AugmentationEnum.AUTOAUG: AutoAugment,
    AugmentationEnum.NONE: DoNothing
}