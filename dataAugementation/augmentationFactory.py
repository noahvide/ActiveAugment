from typing import List, Optional

from configSetup.configModel import AugmentationItem
from dataAugementation.augmentations.doNothing import DoNothing
from dataAugementation.intensity import Intensity
from dataAugementation.augmentations import AUGMENTATION_MAP, AugmentationEnum, BaseAugmentation
from dataProcessing.dataPath import DataPath


def augmentation_factory(aug_yaml_list: Optional[List[AugmentationItem]], fixed_pipeline_length: bool, dataset_name: DataPath) -> List[BaseAugmentation]:
    """
    Parses a list of dicts from YAML and returns a List of BaseTransform.
    Example YAML input: [{'name': 'RotationTransform', 'intensities': ['HIGH', 'LOW']}]
    """
    augmentation_space : List[BaseAugmentation] = []
    if not aug_yaml_list:
        return []
    for entry in aug_yaml_list:
        name = entry.name
        
        # legacy support
        if not name or str(name).lower() == "donothing":
            continue
        
        intensities : Optional[List[Intensity]] = entry.intensities
        if name in AUGMENTATION_MAP:
            if name in [AugmentationEnum.FLIP, AugmentationEnum.TRIVIALAUG]:
                augmentation_space.append(AUGMENTATION_MAP[name]()) # type: ignore
            elif name == AugmentationEnum.AUTOAUG:
                augmentation_space.append(AUGMENTATION_MAP[name](dataset_name=dataset_name)) # type: ignore
            elif intensities:
                for intensity in intensities:
                    augmentation_space.append(AUGMENTATION_MAP[name](intensity))
        else:
            print(f"WARNING: Could not load unknown augmentation {name}")
            
    if not fixed_pipeline_length:
        augmentation_space.append(DoNothing())
    
    return augmentation_space