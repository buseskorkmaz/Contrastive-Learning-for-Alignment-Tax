import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datasets import load_dataset, Dataset
from .toxicity import ToxicityScorer
from .faithfulness import FaithfulnessScorer
from .augmentation import ContrastiveAugmenter
import json
import torch
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    cache_dir: str
    toxicity_threshold: float = 0.5
    faithfulness_threshold: float = 0.7
    max_length: int = 512
    min_length: int = 10
    batch_size: int = 32

class DataProcessor:
    """Main class for processing and preparing the contrastive dataset."""
    
    def __init__(
        self,
        config: ProcessingConfig,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        augmenter: bool = True,
    ):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        device='cpu'
        print("data processor DEVICE:", device)
        # Initialize scorers
        self.toxicity_scorer = ToxicityScorer(device=device)
        print("Toxicity Scorer Initialized.", flush=True)
        self.faithfulness_scorer = FaithfulnessScorer(device=device)
        print("Faithfulness Scorer Initialized.", flush=True)
        if augmenter:
            self.augmenter = ContrastiveAugmenter(device=device)
        else:
            self.augmenter = None
        print("Augmenter Initialized.", self.augmenter, flush=True)
        # Create cache directories
        self.scores_cache = self.cache_dir / "scores"
        self.scores_cache.mkdir(exist_ok=True)
        print("Data Processor Initialized.", flush=True)

    def preprocess_text(self, examples: Dict) -> Dict:
        """Basic text preprocessing."""
        mask = [
            self.config.min_length <= len(text.split()) <= self.config.max_length
            for text in examples["summary"]
        ]
        
        return {
            "post": [x for x, m in zip(examples["post"], mask) if m],
            "summary": [x for x, m in zip(examples["summary"], mask) if m]
        }

    def load_cached_scores(self, batch_id: int) -> Optional[Tuple[List[float], List[float]]]:
        """Load cached scores if they exist."""
        cache_file = self.scores_cache / f"batch_{batch_id}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return (
                    [sample['toxicity'] for sample in data['samples']],
                    [sample['faithfulness'] for sample in data['samples']]
                )
        return None

    def save_scores_cache(self, batch_id: int, examples: Dict, toxicity_scores: List[float], 
                         faithfulness_scores: List[float]):
        """Save scores to cache."""
        cache_file = self.scores_cache / f"batch_{batch_id}.json"
        data = {
            "samples": [
                {
                    "post": post,
                    "summary": summary,
                    "toxicity": tox,
                    "faithfulness": faith
                }
                for post, summary, tox, faith in zip(
                    examples["post"],
                    examples["summary"],
                    toxicity_scores,
                    faithfulness_scores
                )
            ]
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def score_samples(
        self, 
        examples: Dict,
        batch_id: Optional[int] = None
    ) -> Tuple[List[float], List[float]]:
        print("In function score_samples", flush=True)
        """Score samples for toxicity and faithfulness with caching."""
        if batch_id is not None:
            cached_scores = self.load_cached_scores(batch_id)
            if cached_scores:
                return cached_scores
            
        # Early return for non-main processes
        # if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        #     print("Early return for non-main processes", flush=True)
        #     return [0.0] * len(examples["post"]), [0.0] * len(examples["post"])

        print("Scoring samples for toxicity...", flush=True)
        toxicity_scores = self.toxicity_scorer.score_batch(
            examples["summary"],
            batch_size=self.config.batch_size
        )
        
        print("Scoring samples for faithfulness...", flush=True)
        faithfulness_scores = self.faithfulness_scorer.score_batch(
            sources=examples["post"],
            summaries=examples["summary"],
            batch_size=self.config.batch_size
        )
        
        if batch_id is not None:
            self.save_scores_cache(batch_id, examples, toxicity_scores, faithfulness_scores)
        
        return toxicity_scores, faithfulness_scores

    def label_samples(
        self,
        toxicity_scores: List[float],
        faithfulness_scores: List[float]
    ) -> List[int]:
        """Label samples as positive or negative based on scores."""
        labels = []
        for tox, faith in zip(toxicity_scores, faithfulness_scores):
            if tox <= self.config.toxicity_threshold and faith >= self.config.faithfulness_threshold:
                labels.append(1)  # Positive example
            elif tox > self.config.toxicity_threshold or faith < self.config.faithfulness_threshold:
                labels.append(0)  # Negative example
            else:
                labels.append(-1)  # Filtered out
                
        return labels

    def process_augmentation(
        self,
        examples: Dict,
        labels: List[int],
        aug_type: str,
        rank_info: Dict,
        batch_idx: int,
        batch_size: int
    ) -> Dict:
        """Process a specific augmentation type."""
        augmented_samples = []
        rank = rank_info["rank"]
        samples_per_gpu = rank_info["samples_per_gpu"]

        for idx, (post, summary, label) in enumerate(zip(
            examples["post"],
            examples["summary"],
            labels
        )):
            global_idx = rank * samples_per_gpu + batch_idx * batch_size + idx
            aug_method = aug_type.split("/")[1]
            
            try:
                # Handle positive augmentations
                if aug_type == "pos/back_translation" and label == 1:
                    pos_augs = self.augmenter.generate_positive(summary, post, global_idx)
                    if pos_augs:  # Check if we got any augmentations
                        augmented_samples.extend(pos_augs)
                
                # Handle negative augmentations
                elif aug_type.startswith("neg/") and label == 1:
                    neg_augs = self.augmenter.generate_negative(summary, post, aug_method, global_idx)
                    if aug_method in neg_augs:
                        augmented_samples.append(neg_augs)
                else:
                    augmented_samples.append({aug_method: [{
                        "post": post,
                        "summary": summary,
                        "original_idx": global_idx,
                        "augmented": False,
                    }]})
            
            except Exception as e:
                print(f"Error processing example {global_idx}: {e}")
                continue
        
        return {
            "samples": augmented_samples,
            "aug_type": aug_type,
            "total_processed": len(examples["post"]),
            "augmentations_generated": len(augmented_samples)
        }

    def process_batch(self, inputs: Dict, batch_id: int, aug_type: str) -> Dict:
        """Process a batch of data for a specific augmentation type."""
        # Preprocess
        processed = self.preprocess_text(inputs)
        
        # Score samples (with caching)
        toxicity_scores, faithfulness_scores = self.score_samples(processed, batch_id)
        
        # Label samples
        labels = self.label_samples(toxicity_scores, faithfulness_scores)
        
        # Process specific augmentation
        results = self.process_augmentation(processed, labels, aug_type)
        
        return results
    