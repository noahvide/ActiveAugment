from typing import List

import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from configSetup.enums import MetricName


class MetricFactory:
    @staticmethod
    def get_metrics(metric_names: List[MetricName], num_classes: int, device: torch.device):
        METRICS_MAP = {
            MetricName.ACCURACY: MulticlassAccuracy(num_classes=num_classes),
            MetricName.F1_SCORE: MulticlassF1Score(num_classes=num_classes, average="macro"),
            MetricName.PRECISION: MulticlassPrecision(num_classes=num_classes, average="macro"),
            MetricName.RECALL: MulticlassRecall(num_classes=num_classes, average="macro")
        }
        selected_metrics = {}
        for name in metric_names:
            if name in METRICS_MAP:
                selected_metrics[name.value] = METRICS_MAP[name]
            else:
                print(f"WARNING: Could not load unknown metric {name}")
        
        return torchmetrics.MetricCollection(selected_metrics).to(device)