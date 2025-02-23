import os
from pathlib import Path
import json
import logging
from datasets import Dataset
import glob
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_pos_neg_samples(
    samples: list,
    toxicity_threshold: float,
    faithfulness_threshold: float
) -> tuple:
    """Split samples into positive and negative based on scores."""
    positive_samples = []
    negative_samples = []
    
    for sample in samples:
        if sample['toxicity'] <= toxicity_threshold and sample['faithfulness'] >= faithfulness_threshold:
            positive_samples.append(sample)
        elif sample['toxicity'] > toxicity_threshold or sample['faithfulness'] < faithfulness_threshold:
            negative_samples.append(sample)
    
    return positive_samples, negative_samples

def load_all_cached_scores(cache_dir: Path, raw_data_path: str = None) -> list:
    """
    Load all cached scoring results while maintaining batch organization.
    """
    all_scores = []
    cache_files = sorted(glob.glob(str(cache_dir / "batch_*.json")))
    
    logger.info(f"Found {len(cache_files)} cached score files")
    
    for cache_file in tqdm(cache_files, desc="Loading cached scores"):
        batch_id = int(Path(cache_file).stem.split('_')[1])  # Get batch number from filename
        
        with open(cache_file, 'r') as f:
            try:
                score_data = json.load(f)
                
                if not isinstance(score_data, dict) or 'samples' not in score_data:
                    logger.warning(f"Invalid format in {cache_file}")
                    continue
                
                # Extract scores and combine with original data
                for sample in score_data['samples']:
                    if isinstance(sample, dict) and all(k in sample for k in ['post', 'summary', 'toxicity', 'faithfulness']):
                        processed_sample = {
                            'post': sample['post'],
                            'summary': sample['summary'],
                            'toxicity': sample['toxicity'],
                            'faithfulness': sample['faithfulness'],
                            'batch_id': batch_id  # Keep track of which batch this came from
                        }
                        all_scores.append(processed_sample)
                    else:
                        logger.warning(f"Skipping malformed sample in batch {batch_id}")
                
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {cache_file}")
                continue
            except Exception as e:
                logger.warning(f"Error processing {cache_file}: {str(e)}")
                continue
    
    # Sort by batch_id to maintain original order
    all_scores.sort(key=lambda x: x['batch_id'])
    
    logger.info(f"Successfully loaded {len(all_scores)} samples from {len(cache_files)} batches")
    return all_scores

def save_dataset(samples: list, output_dir: Path, split_name: str):
    """Convert samples to dataset and save with batch information."""
    dataset_dict = {
        "post": [s['post'] for s in samples],
        "summary": [s['summary'] for s in samples],
        "toxicity": [s['toxicity'] for s in samples],
        "faithfulness": [s['faithfulness'] for s in samples],
        "batch_id": [s['batch_id'] for s in samples]  # Keep batch information
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    output_path = output_dir / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    
    # Save metadata
    metadata = {
        "split": split_name,
        "total_samples": len(samples),
        "num_batches": len(set(s['batch_id'] for s in samples)),
        "avg_toxicity": sum(s['toxicity'] for s in samples) / len(samples),
        "avg_faithfulness": sum(s['faithfulness'] for s in samples) / len(samples),
        "samples_per_batch": {
            str(batch_id): len([s for s in samples if s['batch_id'] == batch_id])
            for batch_id in sorted(set(s['batch_id'] for s in samples))
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(samples)} samples from {metadata['num_batches']} batches to {output_path}")
    logger.info(f"Average toxicity: {metadata['avg_toxicity']:.3f}")
    logger.info(f"Average faithfulness: {metadata['avg_faithfulness']:.3f}")
    logger.info(f"Samples per batch: min={min(metadata['samples_per_batch'].values())}, "
               f"max={max(metadata['samples_per_batch'].values())}")

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Setup paths
    cache_dir = Path("constrastove_training/data/scores/cache")
    output_base_dir = Path(cfg.data.output_path) / "original"
    
    logger.info("Loading cached scores...")
    all_samples = load_all_cached_scores(cache_dir)
    logger.info(f"Loaded {len(all_samples)} total samples")
    
    # Split into positive and negative samples
    pos_samples, neg_samples = split_pos_neg_samples(
        all_samples,
        toxicity_threshold=cfg.processing.toxicity_threshold,
        faithfulness_threshold=cfg.processing.faithfulness_threshold
    )
    
    logger.info(f"Found {len(pos_samples)} positive and {len(neg_samples)} negative samples")
    logger.info(f"Positive samples from {len(set(s['batch_id'] for s in pos_samples))} batches")
    logger.info(f"Negative samples from {len(set(s['batch_id'] for s in neg_samples))} batches")
    
    # Save datasets
    logger.info("Saving positive samples...")
    save_dataset(pos_samples, output_base_dir, "pos")
    
    logger.info("Saving negative samples...")
    save_dataset(neg_samples, output_base_dir, "neg")

if __name__ == "__main__":
    main()