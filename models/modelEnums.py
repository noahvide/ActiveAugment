
from enum import Enum


class ModelArchitecture(str, Enum):
    RESNET18 = "resnet18"
    TINYVIT = "tinyvit"

class WeightsStatus(str, Enum):
    FROZEN = "frozen"
    TUNABLE = "tunable"
    RANDOM = "random"