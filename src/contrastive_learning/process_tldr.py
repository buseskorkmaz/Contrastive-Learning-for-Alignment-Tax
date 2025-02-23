import os
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from datasets import load_from_disk, Dataset
from typing import List, Dict
from tqdm.auto import tqdm
import json
import warnings
import glob
warnings.filterwarnings('ignore')

from src.data import (
    DataProcessor,
    ProcessingConfig,
    ContrastiveAugmenter
)

logger = logging.getLogger(__name__)

# def setup_distributed():
#     """Setup distributed training."""
#     if "LOCAL_RANK" in os.environ:
#         local_rank = int(os.environ["LOCAL_RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
        
#         dist.init_process_group(
#             backend="nccl",
#             init_method="env://"
#         )
        
#         torch.cuda.set_device(local_rank)
#         return local_rank, world_size
#     return 0, 1

import os
import torch.distributed as dist
import torch

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


# def setup_distributed():
#     if "LOCAL_RANK" in os.environ:
#         local_rank = int(os.environ["LOCAL_RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])

#         # Example: each rank uses 2 GPUs
#         # rank 0 => devices 0,1
#         # rank 1 => devices 2,3
#         # rank 2 => devices 4,5
#         # rank 3 => devices 6,7
#         gpus_per_process = 2
#         start_gpu = local_rank * gpus_per_process
#         end_gpu = start_gpu + gpus_per_process - 1
#         gpu_ids = list(range(start_gpu, end_gpu + 1))
#         gpu_list = ",".join(str(gid) for gid in gpu_ids)
#         os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

#         # Now init
#         dist.init_process_group(backend="nccl", init_method="env://")
#         # Each process sees only its slice of GPUs
#         # So from the process's perspective, it sees 'gpus_per_process' devices
#         # at local indices 0..(gpus_per_process-1)
#         torch.cuda.set_device(0)  # pick device index 0 in this smaller visible set
#         return local_rank, world_size

#     return 0, 1


def load_cached_scores(cache_dir: Path, batch_id: int) -> Dict:
    """Load cached scoring results."""
    cache_file = Path("constrastove_training/data/scores/cache") / f"batch_{batch_id}.json"
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return None

def save_cached_scores(cache_dir: Path, batch_id: int, scores_data: Dict):
    """Save scoring results to cache."""
    cache_file = cache_dir / f"batch_{batch_id}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(scores_data, f)

def gather_and_save_results(
    output_dir: Path,
    aug_type: str,
    world_size: int,
    final_output_path: str,
    is_main_process: bool,
    samples_per_gpu: int,
    is_original: bool = False
) -> None:
    
    if is_main_process:
        logger.info("Gathering results from all ranks...")
        all_samples = []
        aug_method = aug_type.split("/")[1]
        print("aug_method", aug_method)
        
        for rank in range(world_size):
            rank_dir = output_dir / aug_type / f"rank_{rank}"
            batch_files = sorted(glob.glob(str(rank_dir / "batch_*.json")))
            
            for batch_file in batch_files:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    all_samples.extend(batch_data['samples'])
        
        logger.info(f"Total samples before processing: {len(all_samples)}")
        
        # Convert to dataset format
        dataset_dict = {
            "post": [],
            "summary": [],
            "original_idx": [],
            "augmented": []
        }
        
        if is_original:
            # For original samples, handle the nested structure
            for sample in all_samples:
                if 'original' in sample and isinstance(sample['original'], list):
                    for orig_data in sample['original']:
                        if all(k in orig_data for k in ['post', 'summary', 'original_idx']):
                            dataset_dict["post"].append(orig_data["post"])
                            dataset_dict["summary"].append(orig_data["summary"])
                            dataset_dict["original_idx"].append(orig_data["original_idx"])
                            dataset_dict["augmented"].append(orig_data.get("augmented", False))
        else:
            # Process augmented samples
            aug_method = aug_type.split("/")[1]
            valid_samples = []
            for sample in all_samples:
                if aug_method in sample:
                    inner_samples = sample[aug_method]
                    if isinstance(inner_samples, list):
                        valid_samples.extend(inner_samples)
            
            logger.info(f"Found {len(valid_samples)} samples before sorting")
            
            valid_samples.sort(key=lambda x: x['original_idx'])
            
            for sample in valid_samples:
                if all(k in sample for k in ['post', 'summary', 'original_idx']):
                    dataset_dict["post"].append(sample["post"])
                    dataset_dict["summary"].append(sample["summary"])
                    dataset_dict["original_idx"].append(sample["original_idx"])
                    dataset_dict["augmented"].append(sample.get("augmented", False))
        
        logger.info(f"Final processed samples: {len(dataset_dict['original_idx'])}")
        
        # Verify indices
        indices = dataset_dict["original_idx"]
        if len(indices) > 0:
            logger.info(f"Index range: {min(indices)} to {max(indices)}")
            if not is_original and not all(i+1 == j for i, j in zip(indices, indices[1:])):
                logger.warning("Warning: Indices are not continuous!")
        
        # Save dataset
        final_output_dir = Path(final_output_path) / aug_type
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = Dataset.from_dict(dataset_dict)
        dataset.save_to_disk(str(final_output_dir))
        
        # Save metadata with additional info
        metadata = {
            "aug_type": aug_type,
            "total_samples": len(dataset),
            "num_augmented": sum(dataset_dict["augmented"]),
            "num_original": len(dataset_dict["augmented"]) - sum(dataset_dict["augmented"]),
            "distribution_info": {
                "world_size": world_size,
                "samples_per_gpu": samples_per_gpu
            }
        }
        
        with open(final_output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

def process_batch(
    batch_data: Dict,
    processor: DataProcessor,
    device: torch.device,
    aug_type: str,
    cache_dir: Path,
    batch_id: int,
    rank_info: Dict,
    batch_idx: int,
    batch_size: int
) -> Dict:
    """Process a batch of data for a specific augmentation type."""
    with torch.cuda.device(device):
        # Create input format
        inputs = {
            "post": [item['content'] for item in batch_data],
            "summary": [item['summary'] for item in batch_data]
        }
        
        # Check cache for scores
        # cached_scores = load_cached_scores(cache_dir, batch_id)
        # if cached_scores:
        #     toxicity_scores = [s['toxicity'] for s in cached_scores['samples']]
        #     faithfulness_scores = [s['faithfulness'] for s in cached_scores['samples']]
        # else:
        # Calculate and cache scores
        toxicity_scores, faithfulness_scores = processor.score_samples(inputs)
        scores_data = {
            "samples": [
                {
                    "post": post,
                    "summary": summary,
                    "toxicity": tox,
                    "faithfulness": faith
                }
                for post, summary, tox, faith in zip(
                    inputs["post"],
                    inputs["summary"],
                    toxicity_scores,
                    faithfulness_scores
                )
            ]
        }
        save_cached_scores(cache_dir, batch_id, scores_data)
            
        # Process specific augmentation
        labels = processor.label_samples(toxicity_scores, faithfulness_scores)
        results = processor.process_augmentation(
            inputs, labels, aug_type, rank_info, batch_idx, batch_size
        )
        
        return results
        
@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # First, setup distributed processing
    local_rank, world_size = setup_distributed()
    is_main_process = (local_rank == 0)
    device = torch.device("cuda", local_rank)
    # local_rank, world_size = setup_distributed()  # always 0,1
    # is_main_process = True
    # device = torch.device("cuda")  # or "cuda" if you prefer
    
    if is_main_process:
        logger.info(f"\nConfig:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"Starting processing with {world_size} GPUs")
    
    # Get augmentation type from config
    aug_type = cfg.augmentation.type
    is_original = aug_type == "pos/original" 
    logger.info(f"Processing augmentation type: {aug_type}")
    
    # Create output directory using config path
    base_output_dir = Path(str(cfg.output.dir))
    output_dir = base_output_dir / aug_type
    cache_dir = base_output_dir / "cache"
    
    # Create rank-specific output directory
    rank_output_dir = output_dir / f"rank_{local_rank}"
    rank_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    process_config = ProcessingConfig(
        toxicity_threshold=cfg.processing.toxicity_threshold,
        faithfulness_threshold=cfg.processing.faithfulness_threshold,
        batch_size=cfg.data.process_batch_size,
        cache_dir=str(cache_dir)
    )
    processor = DataProcessor(process_config)
    
    # Load dataset
    if is_main_process:
        logger.info("Loading dataset...")
    
    dataset = load_from_disk(f"{cfg.data.path}/{cfg.split}")
    
    # Adjust num_samples if it's larger than dataset size
    total_samples = len(dataset)
    cfg.data.num_samples = min(cfg.data.num_samples, total_samples)
    
    # Calculate indices for distribution
    samples_per_gpu = cfg.data.num_samples // world_size
    start_idx = local_rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if local_rank < world_size - 1 else cfg.data.num_samples
    actual_samples = end_idx - start_idx
    dataset = dataset.select(range(start_idx, end_idx))
    
    # Process batches
    logger.info(f"Rank {local_rank}: Processing {actual_samples} samples for {aug_type}...")
    logger.info(f"Rank {local_rank}: Index range: {start_idx} to {end_idx}")
   
    results = []
    current_batch = []
    batch_counter = 0
    
    for i, example in enumerate(tqdm(dataset, total=actual_samples, disable=not is_main_process)):
        current_batch.append(example)
        
        if len(current_batch) >= cfg.data.batch_size or i == actual_samples - 1:
            if len(current_batch) > 0:
                batch_results = process_batch(
                    batch_data=current_batch,
                    processor=processor,
                    device=device,
                    aug_type=aug_type,
                    cache_dir=cache_dir,
                    batch_id=start_idx + batch_counter,
                    rank_info={
                        "start_idx": start_idx,
                        "rank": local_rank,
                        "samples_per_gpu": samples_per_gpu
                    },
                    batch_idx=batch_counter,
                    batch_size=cfg.data.batch_size
                )

                # Save results with rank info
                batch_file = rank_output_dir / f"batch_{batch_counter}.json"
                with open(batch_file, 'w') as f:
                    json.dump(batch_results, f)
                
                results.append(batch_file)
                current_batch = []
                batch_counter += 1
    
    # Synchronize processes
    if world_size > 1:
        dist.barrier()
    
    # Gather and save results with distribution info
    gather_and_save_results(
        output_dir=base_output_dir,
        aug_type=aug_type,
        world_size=world_size,
        final_output_path=cfg.data.output_path,
        is_main_process=is_main_process,
        samples_per_gpu=samples_per_gpu,
        is_original=is_original 
    )
    
    if is_main_process:
        logger.info(f"All ranks completed processing for {aug_type}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()