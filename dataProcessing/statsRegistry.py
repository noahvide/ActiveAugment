import yaml
from dataProcessing.dataPath import DataPath


def get_dataset_stats(datapath: DataPath, yaml_path: str = "./dataProcessing/dataset_stats.yaml"):
    try:
        with open(yaml_path, 'r') as f:
            external_stats = yaml.safe_load(f)
            return external_stats[datapath.name]
    except (FileNotFoundError, KeyError):
        raise RuntimeError("Could not load stats for dataset_name")