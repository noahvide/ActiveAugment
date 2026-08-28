import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dataProcessing.dataPath import DataPath

def calculate_dataset_stats(root_path: DataPath, img_size=(256, 256), batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(), 
    ])

    data_path = Path(root_path.value) / "train" 
    
    if not data_path.exists():
        print(f"Skipping {root_path.name}: Path {data_path} does not exist.")
        return None, None

    dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    cnt = 0
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)

    print(f"\nCalculating stats for {root_path.name}...")
    for images, _ in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment ** 2)

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    output_file = Path("dataset_stats.yaml")
    
    if output_file.exists():
        with open(output_file, 'r') as f:
            all_stats = yaml.safe_load(f) or {}
    else:
        all_stats = {}

    for root_dir in DataPath:
        mean, std = calculate_dataset_stats(root_dir, batch_size=32)
        
        if mean and std:
            all_stats[root_dir.name] = {
                "mean": mean,
                "std": std
            }
            print(f"Done: {root_dir.name}")

    with open(output_file, 'w') as f:
        yaml.dump(all_stats, f, default_flow_style=False)
    
    print(f"\nResults saved to {output_file.absolute()}")