import os
from pathlib import Path
from typing import Iterator

import torch

current_dir = Path(__file__).resolve().parent
weights_dir = current_dir / "models_weights"
weights_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(weights_dir)
os.environ["HF_HOME"] = str(weights_dir)
os.environ["HF_HUB_OFFLINE"] = "1"

from models.modelEnums import ModelArchitecture, WeightsStatus
import torch.nn as nn
import torchvision.models as models
import timm

from typing import Iterator
import torch
import torch.nn as nn
import torchvision.models as models
import timm

class ALModel(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        drop_rate = 0.5
        self.dropout = nn.Dropout(p=drop_rate)
        
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor, stochastic: bool = False) -> torch.Tensor:
        features = self._encode(x)
        if stochastic:
            features = self._dropout(features) 
        return self._classify(features)
        
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def _dropout(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(features)
    
    def _classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)
        
    def _clf_parameters(self) -> Iterator[nn.Parameter]:
        return self.classifier.parameters()


def get_model(model_name: ModelArchitecture, num_classes: int, weights_status: WeightsStatus) -> ALModel:
    """
    Factory function to fetch models.
    Sets up architecture for both 'from scratch' and 'finetuning'.
    """    
    pretrained = weights_status in ["frozen", "tunable"]
    
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)        
        
        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity() # type: ignore
        
    elif model_name == "tinyvit":
        backbone = timm.create_model(
            "tiny_vit_11m_224", 
            pretrained=pretrained, 
            num_classes=0 
        )
        
        feature_dim = backbone.num_features
        
    else:
        raise ValueError(f"Model {model_name} not supported in this framework.")

    if weights_status == "frozen":
        for param in backbone.parameters():
            param.requires_grad = False
            

    model = ALModel(
        backbone=backbone,
        feature_dim=feature_dim, # type: ignore
        num_classes=num_classes,
    )
    
    return model
