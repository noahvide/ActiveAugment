from enum import Enum
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

class DataPath(str, Enum):
    CIFAR_10 = "CIFAR_10"
    CIFAR_100 = "CIFAR_100"
    MNIST = "MNIST"
    STL = "STL"
    BRISC = "BRISC"
    BUSI = "BUSI"
    FETAL_PLANES = "FETAL_PLANES"
    ISIC = "ISIC"
    
    @property
    def get_path(self) -> str:
        paths = {
            "CIFAR_10": str(PROJECT_ROOT / "data/Natural/cifar-10"),
            "CIFAR_100": str(PROJECT_ROOT / "data/Natural/cifar-100"),
            "MNIST": str(PROJECT_ROOT / "data/Natural/MNIST"),
            "STL": str(PROJECT_ROOT / "data/Natural/stl-10"),
            "BRISC": str(PROJECT_ROOT / "data/Medical/BRISC2025"),
            "BUSI": str(PROJECT_ROOT / "data/Medical/BUSI"),
            "FETAL_PLANES": str(PROJECT_ROOT / "data/Medical/FETAL-PLANES-DB"),
            "ISIC": str(PROJECT_ROOT / "data/Medical/ISIC-2019")
        }
        return paths[self.value]