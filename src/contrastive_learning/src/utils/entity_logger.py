import spacy
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Set, Union, Tuple
from datasets import load_from_disk, Dataset, concatenate_datasets
import logging
from tqdm import tqdm
from dataclasses import dataclass
import argparse

@dataclass
class EntityAnalysis:
    """Store entity analysis results for a sample pair."""
    source_entities: List[str]
    target_entities: List[str]
    entity_tokens: List[int]
    aug_type: str

class EntityLogger:
    """Logger for analyzing and tracking named entities across augmentation types."""
    
    def __init__(self, spacy_model: str = "en_core_web_trf"):
        self.nlp = spacy.load(spacy_model)
        self.entity_stats = defaultdict(lambda: {
            "total_occurrences": 0,
            "unique_entities": set(),
            "entity_types": defaultdict(int)
        })
    
    def _get_entity_tokens(self, doc: spacy.tokens.Doc) -> List[int]:
        """Create binary entity markers."""
        words = doc.text.split()
        tokens = [0] * len(words)
        for ent in doc.ents:
            start_word = len(doc.text[:ent.start_char].split())
            end_word = len(doc.text[:ent.end_char].split())
            for i in range(start_word, end_word):
                tokens[i] = 1
        return tokens + [0]  # Add EOS marker
    
    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[str]:
        """Extract entity texts."""
        return [ent.text for ent in doc.ents]
    
    def _serialize_stats(self, stats: Dict) -> Dict:
        """Convert stats dictionary to JSON-serializable format."""
        return {
            "total_occurrences": stats["total_occurrences"],
            "unique_entities": list(stats["unique_entities"]),
            "entity_types": dict(stats["entity_types"]),
            "unique_entity_count": len(stats["unique_entities"])
        }
    
    def process_augmentation_dir(
        self,
        aug_dir: Path,
        output_dir: Path
    ) -> Dict[str, Dataset]:
        """Process all augmentation types in a directory."""
        aug_datasets = {}
        aug_type = aug_dir.name
        
        # Load dataset for this augmentation type
        dataset = load_from_disk(str(aug_dir))
        
        # Group samples by original index
        samples_by_idx = defaultdict(lambda: {
            "text": None,
            "summaries": [],
            "entity_tokens": [],
            "source_entities": None,
            "aug_type": aug_type
        })
        
        # Process samples
        for item in tqdm(dataset, desc=f"Processing {aug_type}"):
            orig_idx = item["original_idx"]
            
            # Process source text if not done yet
            if samples_by_idx[orig_idx]["text"] is None:
                source_doc = self.nlp(item["post"])
                source_entities = self._extract_entities(source_doc)
                samples_by_idx[orig_idx].update({
                    "text": item["post"],
                    "source_entities": source_entities
                })
            
            # Process summary
            summary_doc = self.nlp(item["summary"])
            summary_entities = self._extract_entities(summary_doc)
            entity_tokens = self._get_entity_tokens(summary_doc)
            
            samples_by_idx[orig_idx]["summaries"].append(item["summary"])
            samples_by_idx[orig_idx]["entity_tokens"].append(entity_tokens)
            
            # Update stats
            self.entity_stats[aug_type]["total_occurrences"] += len(summary_entities)
            self.entity_stats[aug_type]["unique_entities"].update(summary_entities)
            for ent in summary_doc.ents:
                self.entity_stats[aug_type]["entity_types"][ent.label_] += 1
        
        # Convert to dataset format
        dataset_dict = {
            "text": [],
            "neg_target": [],
            "neg_ne": [],
            "source_entities": [],
            "aug_type": [],
            "original_idx": []
        }
        
        for idx, data in samples_by_idx.items():
            dataset_dict["text"].append(data["text"])
            dataset_dict["neg_target"].append(data["summaries"])
            dataset_dict["neg_ne"].append(data["entity_tokens"])
            dataset_dict["source_entities"].append(data["source_entities"])
            dataset_dict["aug_type"].append(data["aug_type"])
            dataset_dict["original_idx"].append(idx)
        
        aug_datasets[aug_type] = Dataset.from_dict(dataset_dict)
        
        # Save individual augmentation results
        aug_output_dir = output_dir / aug_type
        aug_output_dir.mkdir(parents=True, exist_ok=True)
        
        aug_datasets[aug_type].save_to_disk(aug_output_dir / "entity_dataset")
        
        # Save stats in JSON-serializable format
        with open(aug_output_dir / "entity_stats.json", 'w') as f:
            json.dump(self._serialize_stats(self.entity_stats[aug_type]), f, indent=2)
        
        return aug_datasets
    
    def process_all_augmentations(
        self,
        base_dir: Path,
        include_pos: bool = False
    ) -> Dataset:
        """Process all augmentation directories and optionally combine with positives."""
        base_dir = Path(base_dir)
        output_dir = base_dir.parent / "entity_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_datasets = []
        
        # Process each augmentation directory
        aug_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        for aug_dir in aug_dirs:
            aug_datasets = self.process_augmentation_dir(aug_dir, output_dir)
            all_datasets.extend(aug_datasets.values())
        
        # Optionally process positives
        if include_pos:
            pos_dir = base_dir.parent / "pos" / "original"
            if pos_dir.exists():
                pos_dataset = load_from_disk(str(pos_dir))
                pos_processed = self._process_positives(pos_dataset)
                all_datasets.append(pos_processed)
        
        # Combine all datasets
        combined_dataset = concatenate_datasets(all_datasets)
        
        # Save combined dataset and stats
        combined_dataset.save_to_disk(output_dir / "combined_entity_dataset")
        
        # Save overall statistics in JSON-serializable format
        overall_stats = {
            "total_samples": len(combined_dataset),
            "augmentation_types": list(self.entity_stats.keys()),
            "stats_per_type": {
                aug_type: self._serialize_stats(stats)
                for aug_type, stats in self.entity_stats.items()
            }
        }
        
        with open(output_dir / "overall_stats.json", 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        return combined_dataset
    
    def _process_positives(self, dataset: Dataset) -> Dataset:
        """Process positive samples."""
        dataset_dict = {
            "text": [],
            "pos_target": [],
            "pos_ne": [],
            "source_entities": [],
            "aug_type": [],
            "original_idx": []
        }
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing positives")):
            source_doc = self.nlp(item["post"])
            source_entities = self._extract_entities(source_doc)
            
            summary_doc = self.nlp(item["summary"])
            entity_tokens = self._get_entity_tokens(summary_doc)
            
            dataset_dict["text"].append(item["post"])
            dataset_dict["pos_target"].append([item["summary"]])
            dataset_dict["pos_ne"].append([entity_tokens])
            dataset_dict["source_entities"].append(source_entities)
            dataset_dict["aug_type"].append("original")
            dataset_dict["original_idx"].append(idx)
        
        return Dataset.from_dict(dataset_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True,
                      help='Base directory containing augmentation folders (processed_v2/neg)')
    parser.add_argument('--include_pos', action='store_true',
                      help='Include positive samples from pos/original')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    entity_logger = EntityLogger()
    
    dataset = entity_logger.process_all_augmentations(
        args.base_dir,
        include_pos=args.include_pos
    )
    
    logger.info(f"Processed dataset with {len(dataset)} samples")

if __name__ == "__main__":
    main()
