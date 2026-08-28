from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torchvision.transforms import v2 
from dataAugementation.intensity import Intensity, IntensityRange


class BaseAugmentation(v2.Transform):
    CONFIG: Dict[Intensity, Union[IntensityRange, Tuple[IntensityRange, ...]]]

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def get_name(cls):
        return cls.__name__



    def __init__(self, value: Union[float, Tuple[float, float], Intensity], **kwargs):
        super().__init__()
        if self.CONFIG:
            flat_low = list(self._flatten(self._prepare_ranges(Intensity.LOW)))
            flat_medium = list(self._flatten(self._prepare_ranges(Intensity.MEDIUM)))
            flat_high = list(self._flatten(self._prepare_ranges(Intensity.HIGH)))
            all_vals = flat_low + flat_medium + flat_high
            self.CONFIG[Intensity.ALL] = IntensityRange(min(all_vals), max(all_vals))
        self.intensity: Optional[Intensity] = None
        self.resolved_ranges = self._prepare_ranges(value)
        self.value = None

    def _prepare_ranges(self, val: Union[float, Tuple[float, float], Intensity]) -> Tuple[Any, ...]:
        if isinstance(val, Intensity):
            self.intensity = val
            config_val = self.CONFIG[val]
            if isinstance(config_val, tuple):
                return tuple((r.MIN, r.MAX) for r in config_val)
            return ((config_val.MIN, config_val.MAX),)

        elif isinstance(val, tuple):
            return ((float(val[0]), float(val[1])),)
        
        else: 
            return ((float(val), float(val)),)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        range_idx = int(torch.randint(0, len(self.resolved_ranges), (1,)).item())
        chosen_range = self.resolved_ranges[range_idx]
        sampled_val = torch.empty(1).uniform_(chosen_range[0], chosen_range[1]).item()
        self.value = sampled_val
        return dict(value=sampled_val)

    def _flatten(self, data):
        if isinstance(data, tuple):
            for x in data:
                yield from self._flatten(x)
        else:
            yield data

    def to_dict(self):
        flat_range = list(self._flatten(self.resolved_ranges))
            
        return {"augmentation": self.get_name(), 
                "intensity": self.intensity.name if self.intensity else "N/A", 
                "min": min(flat_range),
                "max": max(flat_range),
                "sampled_value": self.value}

    def get_range(self) -> str:
        if len(self.resolved_ranges) == 1:
            range_str = f"{self.resolved_ranges[0]}"
        else:
            range_str = f"{self.resolved_ranges}"
        return range_str
        
    def __repr__(self) -> str:
        if self.intensity:
            return f"{self.name}_{self.intensity.name}"
        else:
            return f"{self.name}"
    
    def __str__(self) -> str:
        return self.__repr__()
