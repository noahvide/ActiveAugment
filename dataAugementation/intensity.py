from dataclasses import dataclass
from enum import Enum

import numpy as np
import matplotlib.colors as mcolors

class Intensity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    ALL = "ALL"

@dataclass
class IntensityRange():
    MIN: float
    MAX: float

INTENSITY_MAP = {
    "LOW": Intensity.LOW,
    "MEDIUM": Intensity.MEDIUM,
    "HIGH": Intensity.HIGH,
    "ALL": Intensity.ALL
}

AUGMENTATION_COLORS = {
    'Rotation': '#d62728',       # Red
    'Brightness': '#ff7f0e',     # Orange
    'Contrast': '#2ca02c',       # Green
    'GaussianNoise': '#1f77b4',  # Blue
    'GaussianBlur': '#9467bd',   # Purple
    'Flip': "#d6a332",           # Goldenrod
    'TrivialAugment': "#8c564b", # Chestnut Brown
    'RandAugment': "#008080",    # Deep Teal
    'AutoAugment': "#4b0082",    # Indigo
    'DoNothing': '#909090',      # Neutral Grey
    'N/A': '#909090'
}

def get_intensity_shade(base_color: str, intensity: str) -> str:
    """Tints the base color with white to reduce saturation for lower intensities."""
    if intensity in ['ALL', 'HIGH']:
        return base_color
        
    rgb = np.array(mcolors.to_rgb(base_color))
    white = np.array([1.0, 1.0, 1.0])
    
    if intensity == 'MEDIUM':
        new_rgb = rgb + (white - rgb) * 0.4
    elif intensity == 'LOW':
        new_rgb = rgb + (white - rgb) * 0.7
    else:
        return base_color
        
    return mcolors.to_hex(new_rgb) # type: ignore