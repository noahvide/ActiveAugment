
from typing import List, Optional

from activeLearning.agentStrategy import AgentStrategy
from activeLearning.agents import AGENT_MAP
from configSetup.configModel import ExperimentConfig
from dataAugementation.augmentations.baseAugmentation import BaseAugmentation


def get_agent(config: ExperimentConfig, agent_strategy: Optional[AgentStrategy], augmentation_space: Optional[List[BaseAugmentation]], seed, **kwargs):
    if agent_strategy in AGENT_MAP:
        return AGENT_MAP[agent_strategy](config, seed, augmentation_space, **kwargs)
    else:
        raise ValueError(f"Could not load unknown agent strategy {agent_strategy}")