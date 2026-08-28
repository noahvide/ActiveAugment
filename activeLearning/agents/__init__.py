from typing import Dict, Optional

from activeLearning.agentStrategy import AgentStrategy

from .BaseAgent import BaseAgent
from .RandomAgent import RandomAgent
from .NoAugmentationAgent import NoAugmentationAgent

from .EntropyAgent import EntropyAgent
from .LeastConfidentAgent import LeastConfidentAgent
from .TypiClustAgent import TypiClustAugmentationAgent
from .CoresetAgent import CoresetAgent
from .MarginAgent import MarginAgent
from .BALDAgent import BALDAgent
from .BadgeAgent import BadgeAgent
from .BALAgent import BALAgent

AGENT_MAP : Dict[Optional[AgentStrategy], type[BaseAgent]] = {
    AgentStrategy.RANDOM: RandomAgent,
    AgentStrategy.ENTROPY: EntropyAgent,
    AgentStrategy.LEASTCONFIDENT: LeastConfidentAgent,
    AgentStrategy.TYPICLUST: TypiClustAugmentationAgent,
    AgentStrategy.CORESET: CoresetAgent,
    AgentStrategy.MARGIN: MarginAgent,
    AgentStrategy.BALD: BALDAgent,
    AgentStrategy.BADGE: BadgeAgent,
    AgentStrategy.BAL: BALAgent,
    AgentStrategy.NONE: NoAugmentationAgent,
    None: NoAugmentationAgent
}


