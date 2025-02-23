import os
from datasets import load_dataset
from pathlib import Path
import hydra
from omegaconf import DictConfig

def create_splits(cfg: DictConfig):
    # Create data directories
    data_dir = Path("constrastove_training/data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset(
        "dim/tldr_17_50k",
        split="train",
        trust_remote_code=True
    )

    # Create splits
    dataset = dataset.shuffle(seed=42)
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)

    splits = {
        "train": dataset.select(range(train_size)),
        "valid": dataset.select(range(train_size, train_size + val_size)),
        "test": dataset.select(range(train_size + val_size, total_size))
    }

    # Save splits
    for split_name, split_dataset in splits.items():
        split_dataset.save_to_disk(data_dir / split_name)

@hydra.main(version_base="1.2", config_path="constrastove_training/configs", config_name="config")
def main(cfg: DictConfig):
    create_splits(cfg)

if __name__ == "__main__":
    main()