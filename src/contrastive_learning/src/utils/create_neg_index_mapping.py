import json
import os
from pathlib import Path
import unicodedata
import re
from collections import defaultdict

def normalize_text(text):
    """Normalize text for consistent matching"""
    if not isinstance(text, str):
        return ""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    # Convert to lowercase and replace newlines
    text = text.lower().replace('\n', ' ')
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    return text.strip()

def create_source_mapping(source_file_path):
    """Create mapping from normalized source text to source index"""
    source_map = {}
    with open(source_file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            norm_text = normalize_text(line)
            source_map[norm_text] = idx
    return source_map

def create_augmentation_mapping(data_dir, aug_methods):
    """
    Create mapping files for each augmentation method
    Returns: Dict[str, Dict[int, List[int]]] mapping source_idx -> list of aug_indices
    """
    mappings = {}
    
    # First load and process source file
    source_file = os.path.join(data_dir, "pos", "original", "gpt2", "train", "train.source")
    source_map = create_source_mapping(source_file)
    
    for aug_method in aug_methods:
        print(f"\nProcessing {aug_method}...")
        aug_dir = os.path.join("constrastove_training/data/processed_filled", "neg", aug_method)
        
        if not os.path.exists(aug_dir):
            print(f"Directory not found: {aug_dir}")
            continue
            
        # Load augmented dataset info
        dataset_info_path = os.path.join(aug_dir, "dataset_info.json")
        if not os.path.exists(dataset_info_path):
            print(f"No dataset info found for {aug_method}")
            continue
            
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
            
        # Load augmented data
        data_file = Path(aug_dir)
        if not data_file.exists():
            print(f"No data file found for {aug_method}")
            continue
            
        from datasets import load_from_disk
        table = load_from_disk(str(data_file))
        df = table.to_pandas()
        
        # Create mapping for this augmentation method
        method_mapping = defaultdict(list)
        
        for idx, row in df.iterrows():
            if not row['augmented']:
                continue
                
            norm_post = normalize_text(row['post'])
            if norm_post in source_map:
                source_idx = source_map[norm_post]
                method_mapping[source_idx].append(idx)
        
        # Save mapping for this method
        mapping_file = os.path.join(data_dir, f"{aug_method}_index_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                "method": aug_method,
                "total_sources": len(source_map),
                "total_augmentations": sum(len(v) for v in method_mapping.values()),
                "mapping": {str(k): v for k, v in method_mapping.items()}
            }, f, indent=2)
            
        mappings[aug_method] = method_mapping
        
        # Print statistics
        print(f"Statistics for {aug_method}:")
        print(f"Total sources: {len(source_map)}")
        print(f"Sources with augmentations: {len(method_mapping)}")
        print(f"Total augmentations: {sum(len(v) for v in method_mapping.values())}")
        
    return mappings

# Usage:
data_dir = "constrastove_training/data/processed_v5_debug"
aug_methods = ["mask_ent", "mask_rel", "regen_ent", "regen_rel", "swap_ent"]
mappings = create_augmentation_mapping(data_dir, aug_methods)