from sklearn.model_selection import train_test_split
from typing import List, Dict


def get_cumulative_stratified_indices(full_targets: List[int], budgets: List[int], seed: int) -> Dict[int, List[int]]:
    """Generates cumulative indices ensuring strict subset relations: subset_10 ⊂ subset_20."""
    sorted_budgets = sorted(budgets)
    cumulative_subsets = {}
    
    current_subset = []
    remaining_indices = list(range(len(full_targets)))
    remaining_targets = full_targets.copy()

    for budget in sorted_budgets:
        needed = budget - len(current_subset)
        if needed <= 0:
            cumulative_subsets[budget] = current_subset.copy()
            continue

        if needed >= len(remaining_indices):
            current_subset.extend(remaining_indices)
            cumulative_subsets[budget] = current_subset.copy()
            break

        try:
            selected, remaining_indices, _, remaining_targets = train_test_split(
                remaining_indices, remaining_targets,
                train_size=needed,
                stratify=remaining_targets,
                random_state=seed
            )
        except ValueError:
            raise ValueError(f"Cannot sample {needed} samples when there are {len(remaining_targets)} targets left.")


        current_subset.extend(selected)
        cumulative_subsets[budget] = current_subset.copy()

    return cumulative_subsets