import torch
from torchvision.transforms import v2
from typing import Callable, List, Dict, Optional, Tuple
from copy import deepcopy

from dataAugementation.augmentations.baseAugmentation import BaseAugmentation
from dataAugementation.augmentations.doNothing import DoNothing

class AugmentationService:
    def __init__(self, 
                 augmentations: Optional[List[BaseAugmentation]], 
                 norm_transform: Optional[Callable] = None):

        self.apply_prob = 1.0
        self.pipeline: Optional[v2.Compose] = None
        self.augmentation_space: List[BaseAugmentation] = []
        self.norm_transform = norm_transform
        
        self.update_augmentations(augmentations)
    
    def normalize(self, raw_images: torch.Tensor) -> torch.Tensor:
        if not self.norm_transform:
            raise ValueError("You must provide a norm_transformer before applying augmentation service")
        return self.norm_transform(raw_images)
    
    def set_normalizer(self, norm_transform: Callable):
        self.norm_transform = norm_transform
        
    def update_augmentations(self, new_augmentations: Optional[List[BaseAugmentation]]):
        if not new_augmentations:
            self.pipeline = None
            self.augmentation_space = []
            return
            
        self.augmentation_space = list(new_augmentations)
        self.pipeline = v2.Compose(self.augmentation_space)

    def generate_intensity_sweeps(self, 
                                  raw_images: torch.Tensor,
                                  rng: torch.Generator,
                                  pi_t: Optional[Tuple[float, float, float]] = None) -> Tuple[torch.Tensor, List[List[Dict]]]:
        if not self.norm_transform:
            raise ValueError("You must provide a norm_transformer before applying augmentation service")

        batch_size = raw_images.size(0)
        
        sweeps: Dict[str, List[BaseAugmentation]] = {}
        universal_augs: List[BaseAugmentation] = []

        for aug in self.augmentation_space:
            aug_dict = aug.to_dict()
            intensity = str(aug_dict.get("intensity", "N/A"))
            aug_type = type(aug).__name__

            if aug_type == "DoNothing":
                continue 

            if intensity == "N/A":
                universal_augs.append(aug)
            else:
                if intensity not in sweeps:
                    sweeps[intensity] = []
                sweeps[intensity].append(aug)

        available_intensities = list(sweeps.keys()) if sweeps else ["DEFAULT"]
        if not sweeps:
            sweeps["DEFAULT"] = []

        candidates_list = []
        batch_metadata = []
        
        for i in range(batch_size):
            img = raw_images[i]
            img_candidates = []
            img_meta = []
            
            if pi_t is not None:
                intensity_map = ["LOW", "MEDIUM", "HIGH"]                
                probs = torch.tensor(pi_t, dtype=torch.float)
                idx = int(torch.multinomial(probs, 1, generator=rng).item())
                chosen_intensity = intensity_map[idx]
            else:
                available_intensities = list(sweeps.keys()) if sweeps else ["DEFAULT"]
                idx = int(torch.randint(0, len(available_intensities), (1,), generator=rng).item())
                chosen_intensity = available_intensities[idx]
            
            active_sweep_augs = sweeps[chosen_intensity] + universal_augs
            
            for aug_instance in active_sweep_augs:
                composed_pipeline = v2.Compose([aug_instance])
                aug_img = composed_pipeline(img.clone())
                aug_img = torch.clamp(aug_img, 0.0, 1.0)
                
                img_candidates.append(aug_img)
                img_meta.append({"pipeline": [aug_instance.to_dict()]})

            candidates_list.append(torch.stack(img_candidates))
            batch_metadata.append(img_meta)

        final_candidates = torch.stack(candidates_list)
        
        N, K, C, H, W = final_candidates.shape
        flat_candidates = final_candidates.view(N * K, C, H, W)
        normalized_flat = self.norm_transform(flat_candidates)
        normalized_candidates = normalized_flat.view(N, K, C, H, W)

        return normalized_candidates, batch_metadata
        

    def generate_random_pipelines(self, 
                            raw_images: torch.Tensor, 
                            k: Optional[int], 
                            pipeline_length: int, 
                            rng: torch.Generator,
                            fixed_pipeline_length: bool) -> Tuple[torch.Tensor, List[List[Dict]]]:
        if not self.norm_transform:
            raise ValueError("You must provide a norm_transformer before applying augmentation service")

        if not k:
            k = 8
            

        batch_size = raw_images.size(0)
        candidates_list = []
        batch_metadata = []
        
        aug_space = deepcopy(self.augmentation_space)
        
        donothing_idx = -1
        type_to_indices: Dict[type, List[int]] = {}
        
        for idx, aug in enumerate(aug_space):
            if isinstance(aug, DoNothing):
                donothing_idx = idx
            else:
                aug_type = type(aug)
                if aug_type not in type_to_indices:
                    type_to_indices[aug_type] = []
                type_to_indices[aug_type].append(idx)
                
        if donothing_idx == -1:
            aug_space.append(DoNothing())
            donothing_idx = len(aug_space) - 1
            
        available_types = list(type_to_indices.keys())
        
        for i in range(batch_size):
            img = raw_images[i]
            img_candidates = []
            img_meta = []
            
            for _ in range(k):
                if fixed_pipeline_length:
                    target_length = pipeline_length
                else:
                    target_length = int(torch.randint(0, pipeline_length + 1, (1,), generator=rng).item())
                
                if target_length > len(available_types):
                    raise ValueError(f"Target length {target_length} exceeds number of unique base augmentations ({len(available_types)}).")
                
                active_indices = []
                if target_length > 0:
                    shuffled_type_idx = torch.randperm(len(available_types), generator=rng).tolist()
                    selected_types = [available_types[idx] for idx in shuffled_type_idx[:target_length]]
                    
                    for aug_type in selected_types:
                        variants = type_to_indices[aug_type]
                        chosen_variant = variants[int(torch.randint(0, len(variants), (1,), generator=rng).item())]
                        active_indices.append(chosen_variant)
                
                num_padding = pipeline_length - len(active_indices)
                donothing_indices = [donothing_idx] * num_padding
                ordered_pipeline_indices = active_indices + donothing_indices
                
                transforms_list: List[BaseAugmentation] = []
                pipeline_meta = []

                for aug_idx in ordered_pipeline_indices:
                    aug = aug_space[aug_idx]
                    transforms_list.append(aug)
                    
                    meta_dict = aug.to_dict()
                    meta_dict["space_index"] = aug_idx
                    pipeline_meta.append(meta_dict)

                composed_pipeline = v2.Compose(transforms_list)
                aug_img = composed_pipeline(img.clone())

                aug_img = torch.clamp(aug_img, 0.0, 1.0)

                img_candidates.append(aug_img)
                img_meta.append({"pipeline": pipeline_meta})

            candidates_list.append(torch.stack(img_candidates))
            batch_metadata.append(img_meta)

        final_candidates = torch.stack(candidates_list)
        
        N, K, C, H, W = final_candidates.shape
        flat_candidates = final_candidates.view(N * K, C, H, W)
        normalized_flat = self.norm_transform(flat_candidates)
        normalized_candidates = normalized_flat.view(N, K, C, H, W)

        return normalized_candidates, batch_metadata

    
    def apply_pipeline(self, raw_images: torch.Tensor, aug_pipeline: List[BaseAugmentation], apply_prob = 1.0) -> Tuple[torch.Tensor, List[Dict]]:
        if not self.norm_transform:
            raise ValueError("You must provide a norm_transformer before applying augmentation service")

        batch_size = raw_images.size(0)
        final_images_list = []
        batch_metadata = []
            
        if not aug_pipeline or apply_prob <= 0.0:
            final_images = raw_images
            
            for _ in range(batch_size):
                dummy_pipeline = [{"augmentation": "N/A", "intensity": "N/A", "min": 0.0, "max": 0.0, "sampled_value": 0.0} for _ in aug_pipeline]
                batch_metadata.append([{"pipeline": dummy_pipeline}])
                
        else:
            pipeline = v2.Compose(aug_pipeline)
            mask = torch.rand(batch_size, device=raw_images.device) < apply_prob
            
            for i in range(batch_size):
                img_meta = []
                pipeline_meta = []
                
                if mask[i]:
                    aug_img = pipeline(raw_images[i].clone())
                    aug_img = torch.clamp(aug_img, 0.0, 1.0)
                    final_images_list.append(aug_img)
                    
                    for aug in aug_pipeline:
                        pipeline_meta.append(aug.to_dict())
                else:
                    final_images_list.append(raw_images[i])
                    for aug in aug_pipeline:
                        pipeline_meta.append({"augmentation": "N/A", "intensity": "N/A", "min": 0.0, "max": 0.0, "sampled_value": 0.0})
                
                img_meta.append({"pipeline": pipeline_meta})
                batch_metadata.append(img_meta)
                
            final_images = torch.stack(final_images_list)
        
        normalized_final_images = self.norm_transform(final_images)
        return normalized_final_images, batch_metadata