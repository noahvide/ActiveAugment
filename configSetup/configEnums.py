from enum import Enum


class DatasetSelectionType(str, Enum):
    FULL = "full"
    ACTIVE = "active"

class AugmentationStrategyType(str, Enum):
    NONE = "none"
    STATIC = "static"
    ACTIVE = "active"

class CandidateGenerationMode(str, Enum):
    INTENSITY_SWEEP = "intensity_sweep"
    RANDOM_PIPELINES = "random_pipelines"

class TrainingType(str, Enum):
    STANDARD = "standard"
    CONTRASTIVE = "contrastive"
    MAXUP = "maxup"
