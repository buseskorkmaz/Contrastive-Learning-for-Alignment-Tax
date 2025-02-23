from typing import List, Optional, Dict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass
from .augmentation import (
    ContrastiveAugmenter,
)

@dataclass
class ProcessingConfig:
    toxicity_threshold: float = 0.5
    faithfulness_threshold: float = 0.7
    max_length: int = 512
    min_length: int = 10
    num_augmentations: int = 3
    batch_size: int = 32
    cache_dir: str = "./cache"

class ContrastiveAugmenter:
    """Generate contrastive samples through controlled paraphrasing."""
    
    def __init__(
        self,
        config: ProcessingConfig,
        cache_dir: Optional[str] = None
    ):
        self.augmenter = ContrastiveAugmenter(cache_dir=config.cache_dir)

    def generate_contrastive_samples(
        self,
        examples: Dict,
        labels: List[int],
        rank_info: Dict  # Add rank_info parameter
    ) -> Dict:
        """Generate contrastive samples through controlled paraphrasing."""
        augmented_data = {
            "post": [],
            "summary": [],
            "label": [],
            "original_idx": [],
            "augmentation_type": [],
            "rank_info": []  # Add rank_info tracking
        }
        
        for local_idx, (post, summary, label) in enumerate(zip(
            examples["post"], 
            examples["summary"], 
            labels
        )):
            # Keep original samples
            augmented_data["post"].append(post)
            augmented_data["summary"].append(summary)
            augmented_data["label"].append(label)
            augmented_data["original_idx"].append(local_idx)
            augmented_data["augmentation_type"].append("original")
            augmented_data["rank_info"].append(rank_info)
            
            # Generate positive samples through back translation
            if label == 1:
                positive_samples = self.augmenter.generate_positive(
                    text=summary,
                    source=post,
                    local_idx=local_idx,
                    rank_info=rank_info
                )
                
                for pos_sample in positive_samples:
                    augmented_data["post"].append(pos_sample["post"])
                    augmented_data["summary"].append(pos_sample["summary"])
                    augmented_data["label"].append(1)
                    augmented_data["original_idx"].append(pos_sample["original_idx"])
                    augmented_data["augmentation_type"].append("back_translation")
                    augmented_data["rank_info"].append(rank_info)
            
            # Generate negative samples
            negative_samples = self.augmenter.generate_negative(
                text=summary,
                source=post,
                local_idx=local_idx,
                rank_info=rank_info
            )
            
            for aug_type, samples in negative_samples.items():
                for neg_sample in samples:
                    augmented_data["post"].append(neg_sample["post"])
                    augmented_data["summary"].append(neg_sample["summary"])
                    augmented_data["label"].append(0)
                    augmented_data["original_idx"].append(neg_sample["original_idx"])
                    augmented_data["augmentation_type"].append(aug_type)
                    augmented_data["rank_info"].append(rank_info)
        
        return augmented_data