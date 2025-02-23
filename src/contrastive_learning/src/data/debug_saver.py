import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict

class DataLogger:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.debug_dir = os.path.join(output_dir, "training_data")
        os.makedirs(self.debug_dir, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.batch_counter = 0
        self.sample_counter = 0
        
        # Create an index file to track all samples
        self.index_file = os.path.join(self.debug_dir, "index.jsonl")
        
    def save_batch(self, batch: Dict, dataset_samples: list):
        """Save all samples from each training batch."""
        batch_size = batch["net_input"]["input_ids"].shape[0]
        batch_dir = os.path.join(self.debug_dir, f"batch_{self.batch_counter}")
        os.makedirs(batch_dir, exist_ok=True)
        
        batch_info = {
            "batch_id": self.batch_counter,
            "samples": []
        }
        
        for i in range(batch_size):
            # Get original dataset sample
            dataset_idx = batch["contrast_src_select_index"][i].item()
            orig_sample = dataset_samples[dataset_idx]
            
            sample_info = {
                "id": orig_sample["id"],
                "source": orig_sample["source"],
                "pos_target": orig_sample["pos_target"],
                "pos_ne": orig_sample["pos_ne"],
                "neg_target": orig_sample["neg_target"],
                "neg_ne": orig_sample["neg_ne"],
                "augmentation_types": orig_sample["augmentation_types"],
                "processed": {
                    "input_ids": batch["net_input"]["input_ids"][i].tolist(),
                    "attention_mask": batch["net_input"]["attention_mask"][i].tolist(),
                    "labels": batch["target"][i].tolist(),
                    "ne_mask": batch["contrast_ne"][i].tolist()
                }
            }
            
            # Save individual sample
            sample_path = os.path.join(batch_dir, f"sample_{i}.json")
            with open(sample_path, "w") as f:
                json.dump(sample_info, f, indent=2)
            
            # Add to index
            index_entry = {
                "sample_id": self.sample_counter,
                "batch_id": self.batch_counter,
                "batch_position": i,
                "file_path": os.path.relpath(sample_path, self.debug_dir),
                "source_length": len(orig_sample["source"]),
                "num_positives": len(orig_sample["pos_target"]),
                "num_negatives": len(orig_sample["neg_target"])
            }
            
            with open(self.index_file, "a") as f:
                f.write(json.dumps(index_entry) + "\n")
            
            batch_info["samples"].append(index_entry)
            self.sample_counter += 1
            
        # Save batch metadata
        batch_meta_path = os.path.join(batch_dir, "batch_info.json")
        with open(batch_meta_path, "w") as f:
            json.dump(batch_info, f, indent=2)
            
        self.batch_counter += 1
        
        if self.batch_counter % 100 == 0:
            self.logger.info(f"Logged {self.batch_counter} batches ({self.sample_counter} samples)")