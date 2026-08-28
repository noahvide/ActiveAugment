from enum import Enum


class AgentStrategy(str, Enum):
    RANDOM = "random"
    STATIC = "static"
    ENTROPY = "entropy"
    LEASTCONFIDENT = "least_confident"
    TYPICLUST = "typiclust"
    CORESET = "coreset"
    MARGIN = "margin"
    BALD = "BALD"
    BADGE = "badge"
    BAL = "BAL"
    NONE = "None"

STRATEGY_COLORS = {
    'None': '#333333',
    'random': '#7f7f7f',
    'static': '#607d8b',
    'coreset': '#1f77b4',
    'entropy': '#ff7f0e',
    'least_confident': '#2ca02c',
    'typiclust': '#9467bd',
    'margin': "#e9c837",
    'BALD': "#2cf2f5",
    'badge': "#f2abf2",
    'BAL': "#f2ff63"
}