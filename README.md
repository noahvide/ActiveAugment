# Introduction
This is the repo containing code for the published paper ActiveAugment: Online Active Learning for Augmentation Selection in Deep Learning by Noah Videcrantz and Mostafa Mehdipour Ghazi. The code implements 9 different AL strategy and a framework for exploiting these strategies to select augmented views instead as well as unlabeled data if needed. The project implements 6 different simple augmentations and 3 SOTA augmentations. It implements 2 different backbones and 3 different training regimes. 


# Setup

We recommend using [uv](https://github.com/astral-sh/uv) for fast, deterministic installation. 

```bash
uv sync
```
Dependencies are also available in requirements.txt

For data see README in [data](./data/)

# Structure
[activeLearning/agents](./activeLearning/agents/) contain definitions for AL strategies. 

[configSetup](./configSetup/) deals with defining the config for running experiments. An example of a configuration can be seen in [example_config.yaml](./example_config.yaml). 

[dataAugmentation](./dataAugementation/) contains implementations of the augmentations. 

[dataProcessing](./dataProcessing/) contains methods for working with the data including retrieving the data and splitting the data. [dataset_stats.yaml](./dataProcessing/dataset_stats.yaml) contains stats for the training data used in the paper. If using a different split or different data you need to run [dataset_stats.py](./dataProcessing/dataset_stats.py) to get the updated stats. 

[models](./models/) contain definitions of the backbones used.

[plotting](./plotting/) contains plotting function to monitor the performance of the experiments. In the current state plots are generated during training. 

[run_experiment](./run_experiment/) contains the methods for parsing and executing a configuration including the main routing logic.

[training](./training/) contains definition of different trainers implementing the main training loops used. 




