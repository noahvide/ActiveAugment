from typing import List, Optional, Union, Literal, Annotated
import yaml
from pydantic import BaseModel, Discriminator, Field, model_validator

from configSetup.configEnums import (
    AugmentationStrategyType, 
    CandidateGenerationMode, 
    DatasetSelectionType, 
    TrainingType,
)
from configSetup.enums import DeviceType, MetricName
from models.modelEnums import ModelArchitecture, WeightsStatus
from activeLearning.agentStrategy import AgentStrategy
from dataProcessing.dataPath import DataPath
from dataAugementation.augmentations import AugmentationEnum
from dataAugementation.intensity import Intensity


class DiscrepancyConfig(BaseModel):
    normalize: bool = True

class ModelConfig(BaseModel):
    name: ModelArchitecture
    weights_status: WeightsStatus

class DatasetConfig(BaseModel):
    name: DataPath
    seed: int

class BaseTrainingConfig(BaseModel):
    num_reps: int = Field(gt=0)
    active_epochs: int = Field(gt=0)
    lr: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)
    device: DeviceType
    metrics: List[MetricName]
    monitor: MetricName
    do_evaluation: bool = False

class StandardTrainingConfig(BaseTrainingConfig):
    type: Literal[TrainingType.STANDARD] = TrainingType.STANDARD

class ContrastiveTrainingConfig(BaseTrainingConfig):
    """
    Parameters for Active Invariance Learning.
    Objective: L = L_sup + lambda_con * L_con.
    """
    type: Literal[TrainingType.CONTRASTIVE] = TrainingType.CONTRASTIVE
    lambda_con: float = 0.1  # Loss weighting factor
    temperature: float = 0.07 # Contrastive temperature

class MaxUpTrainingConfig(BaseTrainingConfig):
    type: Literal[TrainingType.MAXUP] = TrainingType.MAXUP

TrainingConfig = Annotated[
    Union[StandardTrainingConfig, AdversarialTrainingConfig, 
          ContrastiveTrainingConfig, MaxUpTrainingConfig],
    Discriminator("type")
]


class AugmentationItem(BaseModel):
    name: AugmentationEnum
    intensities: Optional[List[Intensity]] = None

class AugmentationConfig(BaseModel):
    pipeline_length: int = 1
    fixed_pipeline_length: bool = True
    augmentations: List[AugmentationItem]

class AugmentationStrategyConfig(BaseModel):
    type: AugmentationStrategyType = AugmentationStrategyType.NONE
    agent: Optional[AgentStrategy] = None
    candidate_mode: Optional[CandidateGenerationMode] = None
    k_candidates: Optional[int] = None
    m_select: int = 1
    
    discrepancy: Optional[DiscrepancyConfig] = None
    use_annealing: bool = False
    
    integration_mode: Literal["concat", "fractional"] = "concat"
    integration_fraction: float = 1
    

class DatasetSelectionConfig(BaseModel):
    type: DatasetSelectionType = DatasetSelectionType.FULL
    strategy: Optional[AgentStrategy] = None
    budgets: Optional[List[int]] = None
    continual_learning: bool = False

class ExperimentConfig(BaseModel):
    experiment_name: str
    config_name: str
    
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    
    dataset_selection: DatasetSelectionConfig
    augmentation_strategy: AugmentationStrategyConfig
    augmentation_space: Optional[AugmentationConfig] = None
    
    @model_validator(mode='after')
    def validate_logic(self):
        strat = self.augmentation_strategy
        if self.training.type == TrainingType.MAXUP:
            if strat.type != AugmentationStrategyType.ACTIVE:
                raise ValueError("MaxUp requires 'active' augmentation strategy.")
            if strat.candidate_mode == CandidateGenerationMode.RANDOM_PIPELINES:
                if strat.m_select != strat.k_candidates:
                    raise ValueError(
                        f"For MaxUp with 'random_pipelines', m_select ({strat.m_select}) must equal k_candidates ({strat.k_candidates})."
                    )
            elif strat.candidate_mode == CandidateGenerationMode.INTENSITY_SWEEP:
                if not self.augmentation_space:
                    raise ValueError("MaxUp with 'intensity_sweep' requires an augmentation_space.")
                num_augs = len(self.augmentation_space.augmentations)
                if strat.m_select != num_augs:
                    raise ValueError(
                        f"For MaxUp with 'intensity_sweep', m_select ({strat.m_select}) must equal the number of augmentations in augmentation_space ({num_augs})."
                    )
        
        elif strat.type == AugmentationStrategyType.ACTIVE:
            if not strat.agent:
                raise ValueError("Active Augmentation requires an agent strategy.")
        

        return self

def load_config(yaml_path: str) -> ExperimentConfig:
    with open(yaml_path, 'r') as f:
        raw_dict = yaml.safe_load(f)
    
    return ExperimentConfig(**raw_dict)