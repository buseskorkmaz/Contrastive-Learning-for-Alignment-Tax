import os
import argparse
import json
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict
import spacy
from tqdm import tqdm
import transformers
import random
from collections import defaultdict

def compute_swap_span_bpe(pos_summary, neg_summary, tokenizer):
    """
    Computes the BPE token index where the substitution starts by comparing the BPE-encoded tokens
    of the positive and negative summaries.
    Returns the index in the negative summary where the swap starts.
    """
    # Tokenize the summaries using the tokenizer
    pos_tokens = tokenizer.encode(pos_summary, add_special_tokens=False)
    neg_tokens = tokenizer.encode(neg_summary, add_special_tokens=False)

    # Find the first token index where the tokens differ
    min_len = min(len(pos_tokens), len(neg_tokens))
    swap_start = None
    for idx in range(min_len):
        if pos_tokens[idx] != neg_tokens[idx]:
            swap_start = idx
            break
    if swap_start is None:
        # If all tokens are the same up to min_len, check if lengths differ
        if len(pos_tokens) != len(neg_tokens):
            swap_start = min_len
        else:
            # No difference found
            return None
    return swap_start

def get_entity_tokens(doc):
    """
    Returns a list of integers (0 or 1) indicating whether each token in the doc is part of a named entity.
    """
    tokens = [0] * len(doc)
    for ent in doc.ents:
        for idx in range(ent.start, ent.end):
            tokens[idx] = 1
    # Add an extra 0 for the EOS token
    tokens.append(0)
    return tokens

def verify_data_alignment(source_file, pos_file, neg_file, other_file, tokenizer, num_samples=5):
    """
    Verify alignment between source, positive, and negative samples.
    """
    print("\n=== Data Alignment Verification ===")
    
    # Load files
    with open(source_file) as f_src, \
         open(pos_file) as f_pos, \
         open(neg_file) as f_neg, \
         open(other_file) as f_other:
        
        sources = f_src.readlines()
        pos_samples = f_pos.readlines()
        neg_samples = f_neg.readlines()
        other_info = [line.strip().split('\t') for line in f_other.readlines()]
        
        # Convert other info to dict
        index_map = {i: int(info[0]) for i, info in enumerate(other_info)}
        
        print(f"\nTotal samples:")
        print(f"Sources: {len(sources)}")
        print(f"Positive samples: {len(pos_samples)}")
        print(f"Negative samples: {len(neg_samples)}")
        print(f"Other info: {len(other_info)}")
        
        # Check sample alignments
        print(f"\nChecking {num_samples} random samples:")
        indices = random.sample(range(min(len(sources), len(pos_samples))), num_samples)
        
        for idx in indices:
            print(f"\nSample {idx}:")
            print(f"Original index from mapping: {index_map[idx]}")
            print(f"Source: {sources[idx][:100]}...")
            print(f"Positive: {pos_samples[idx][:100]}...")
            print(f"Negative: {neg_samples[idx][:100]}..." if idx < len(neg_samples) else "No negative sample")
            
            # Verify source matches between original and augmented
            if idx < len(neg_samples):
                source_text = normalize_text(sources[idx])
                orig_idx = index_map[idx]
                if orig_idx >= 0 and orig_idx < len(sources):
                    orig_source = normalize_text(sources[orig_idx])
                    if source_text != orig_source:
                        print("WARNING: Source mismatch!")
                        print(f"Current source: {source_text[:100]}")
                        print(f"Original source: {orig_source[:100]}")

def normalize_text(text):
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = ' '.join(text.split())  # Normalize whitespace
    return text

# Add this check in your process_dataset function:
def check_indices_before_processing(dataset, pos_summaries, index_mapping=None):
    """
    Verify index consistency before processing dataset
    """
    print("\n=== Index Consistency Check ===")
    
    issues = []
    for i, example in enumerate(dataset):
        if isinstance(example, dict):
            original_idx = example.get('original_idx', i)
            
            # Check if using index mapping
            if index_mapping is not None:
                mapped_idx = index_mapping.get(i)
                if mapped_idx is None:
                    issues.append(f"Example {i}: No mapping found")
                    continue
                original_idx = mapped_idx
            
            # Check if original index exists in positive summaries
            if original_idx not in pos_summaries:
                issues.append(f"Example {i}: Original index {original_idx} not in positive summaries")
                continue
                
            # Verify post content matches if available
            if 'post' in example:
                orig_post = normalize_text(example['post'])
                if index_mapping is not None and i in index_mapping:
                    mapped_idx = index_mapping[i]
                    # Here you'd need access to original posts to compare
                    # This is where you might want to add additional verification
                    
    if issues:
        print("\nFound alignment issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(issue)
        print(f"Total issues found: {len(issues)}")
    else:
        print("No alignment issues found")
    
    return len(issues) == 0

def verify_index_alignment(split_name: str, pos_dataset_dir: str, neg_dataset_dir: str, aug_type: str) -> bool:
    """
    Verify alignment between positive and negative examples.
    Returns True if alignment is valid, False otherwise.
    """
    pos_source = []
    neg_source = []
    original_indices = []
    
    # Load source files
    pos_source_path = os.path.join(pos_dataset_dir, f"{split_name}.source")
    neg_source_path = os.path.join(neg_dataset_dir, f"{split_name}.source")
    other_path = os.path.join(neg_dataset_dir, f"{split_name}.other")
    
    try:
        with open(pos_source_path) as f:
            pos_source = [line.strip() for line in f]
        with open(neg_source_path) as f:
            neg_source = [line.strip() for line in f]
        with open(other_path) as f:
            original_indices = [int(line.split('\t')[0]) for line in f]
            
        # Check 1: Original indices should be within bounds
        if max(original_indices) >= len(pos_source):
            print(f"Error: Original index {max(original_indices)} exceeds positive source length {len(pos_source)}")
            return False
            
        # Check 2: Sources should match for corresponding indices
        for neg_idx, orig_idx in enumerate(original_indices):
            if orig_idx >= 0:  # Skip special cases like -1
                if pos_source[orig_idx] != neg_source[neg_idx]:
                    print(f"Error: Source mismatch for negative index {neg_idx} -> original index {orig_idx}")
                    print(f"Pos source: {pos_source[orig_idx][:50]}...")
                    print(f"Neg source: {neg_source[neg_idx][:50]}...")
                    return False
                    
        return True
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

def split_dataset(dataset, splits_ratio, seed=42):
    """
    Splits the dataset while maintaining continuous indexing.
    """
    total = len(dataset)
    train_ratio, validation_ratio, test_ratio = splits_ratio
    
    # Calculate sizes
    train_size = int(total * train_ratio)
    validation_size = int(total * validation_ratio)
    test_size = total - train_size - validation_size
    
    # Create index ranges for each split
    indices = list(range(total))
    random.seed(seed)
    random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]
    
    # Create splits with original indices
    splits = {
        'train': (dataset.select(train_indices), train_indices),
        'validation': (dataset.select(val_indices), val_indices),
        'test': (dataset.select(test_indices), test_indices)
    }
    
    return splits

def create_source_index_mapping(pos_dataset_dir, neg_dataset_dir, split_name):
    """
    Create mapping between negative and positive indices based on source content.
    Includes improved normalization and error handling.
    """
    import unicodedata
    import re
    
    def normalize_text(text):
        """Normalize text consistently."""
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

    # Load positive sources and create mapping
    pos_sources = {}
    pos_source_duplicates = {}
    
    try:
        pos_dataset = load_from_disk(pos_dataset_dir)
        print(f"Loaded positive dataset from {pos_dataset_dir}")
    except Exception as e:
        print(f"Error loading positive dataset: {e}")
        return {}

    # Process positive sources
    for idx, example in enumerate(pos_dataset):
        if 'post' not in example:
            print(f"Warning: Missing 'post' field in positive example {idx}")
            continue
            
        source_text = normalize_text(example['post'])
        if not source_text:
            print(f"Warning: Empty normalized text for positive example {idx}")
            continue
            
        if source_text in pos_sources:
            if source_text not in pos_source_duplicates:
                pos_source_duplicates[source_text] = [pos_sources[source_text]]
            pos_source_duplicates[source_text].append(idx)
            print(f"Warning: Duplicate positive source text found at indices {pos_source_duplicates[source_text]}")
        
        pos_sources[source_text] = idx
        
    print(f"Loaded {len(pos_sources)} unique positive sources")
    if pos_source_duplicates:
        print(f"Found {len(pos_source_duplicates)} duplicate source texts in positive examples")

    # Create mapping for negative examples
    try:
        neg_dataset = load_from_disk(neg_dataset_dir)
        print(f"Loaded negative dataset from {neg_dataset_dir}")
    except Exception as e:
        print(f"Error loading negative dataset: {e}")
        return {}

    index_mapping = {}
    unmapped = []
    stats = {
        'total_neg': 0,
        'found_matches': 0,
        'missing_field': 0,
        'empty_text': 0
    }

    # Process negative examples
    for neg_idx, example in enumerate(neg_dataset):
        stats['total_neg'] += 1
        
        if 'post' not in example:
            stats['missing_field'] += 1
            continue
            
        source_text = normalize_text(example['post'])
        if not source_text:
            stats['empty_text'] += 1
            continue

        if source_text in pos_sources:
            index_mapping[neg_idx] = pos_sources[source_text]
            stats['found_matches'] += 1
        else:
            unmapped.append((neg_idx, source_text))

    # Print detailed statistics
    print("\nMapping Statistics:")
    print(f"Total negative examples: {stats['total_neg']}")
    print(f"Found matches: {stats['found_matches']}")
    print(f"Missing 'post' field: {stats['missing_field']}")
    print(f"Empty normalized text: {stats['empty_text']}")
    print(f"Could not map: {len(unmapped)}")

    if unmapped:
        print("\nFirst few unmapped examples:")
        for idx, text in unmapped[:3]:
            print(f"\nNegative idx {idx}:")
            print(f"Source text: {text[:200]}...")
            # Find closest matches in positive sources for debugging
            print("\nClosest matching positive sources:")
            matches = []
            for pos_text in pos_sources.keys():
                # Simple similarity check - can be improved with better metrics
                if len(set(text.split()) & set(pos_text.split())) / len(set(text.split())) > 0.8:
                    matches.append(pos_text)
            for pos_text in matches[:3]:
                print(f"Positive: {pos_text[:200]}...")

    return index_mapping

def process_dataset(aug_type, dataset_dir, output_dir, tokenizers_dict, nlp, pos_summaries, splits_ratio=(0.8, 0.1, 0.1), seed=42):
    """Process dataset with source-based index alignment and proper BPE handling."""
    if not os.path.exists(dataset_dir):
        print(f"No dataset found in {dataset_dir}, skipping.")
        return

    print(f"\nLoading dataset from: {dataset_dir}")
    try:
        dataset = load_from_disk(dataset_dir)
        print(f"Initial dataset size: {len(dataset)}")
        print("\nDataset fields:", dataset.column_names)
        
        # Debug first few examples
        print("Sample of first few examples:")
        for i, example in enumerate(dataset[:3]):
            print(f"\nExample {i}:")
            if isinstance(example, dict):
                for key, value in example.items():
                    print(f"{key}: {value}")
            else:
                print(f"Value: {example}")
    

        # If we have string data, convert to proper format
        if all(isinstance(example, str) for example in dataset[:10]):
            print("\nConverting string dataset to proper format...")
            features = dataset.features
            if isinstance(features, dict):
                # Create proper dataset structure
                data_dict = {}
                for key in features.keys():
                    data_dict[key] = dataset[key]
                dataset = Dataset.from_dict(data_dict)
                print("Dataset converted successfully")
                print("New fields:", dataset.column_names)
            else:
                print("Warning: Unexpected feature format")
                return
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    splits = split_dataset(dataset, splits_ratio, seed)
    # Initialize augmentation mapping
    augmentation_mapping = defaultdict(list)

    # Create augmentation mapping
    if not aug_type.startswith('pos/'):
        print("\nCreating augmentation mapping...")
        # augmentation_mapping = defaultdict(list)  # source_idx -> list of aug_indices
        
        for i, example in enumerate(dataset):
            if not example.get('augmented', False):
                continue
                
            source_text = normalize_text(example.get('post', ''))
            summary = example.get('summary', '')
            
            # Find matching source
            for src_idx, src_summary in pos_summaries.items():
                src_post = normalize_text(example.get('post', ''))
                if src_post == source_text:
                    augmentation_mapping[src_idx].append({
                        'aug_idx': i,
                        'summary': summary
                    })
                    break
        
        # Save mapping for debugging
        mapping_file = os.path.join(output_dir, f"{aug_type}_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump({
                'method': aug_type,
                'total_sources': len(pos_summaries),
                'total_augmentations': sum(len(v) for v in augmentation_mapping.values()),
                'mapping': {str(k): [x['aug_idx'] for x in v] 
                          for k, v in augmentation_mapping.items()}
            }, f, indent=2)
    
    # Process for each tokenizer
    for tokenizer_name, tokenizer in tokenizers_dict.items():
        tokenizer_output_dir = os.path.join(output_dir, tokenizer_name.split('/')[-1])
        
        for split_name, (split_data, split_indices) in splits.items():
            split_output_dir = os.path.join(tokenizer_output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)

            # Get maximum number of augmentations for any source
            max_augs = max(len(v) for v in augmentation_mapping.values()) if augmentation_mapping else 1
            
            # Create files for each augmentation
            aug_files = []
            for i in range(max_augs):
                prefix = f"{split_name}.{aug_type}.aug{i}"
                aug_files.append({
                    'target': open(os.path.join(split_output_dir, f"{prefix}.neg_target"), 'w'),
                    'ne': open(os.path.join(split_output_dir, f"{prefix}.neg_ne"), 'w'),
                    'other': open(os.path.join(split_output_dir, f"{prefix}.other"), 'w'),
                })

            # Process each source
            for source_idx, example in enumerate(tqdm(split_data)):
                source_text = example.get('post', '')
                
                # Write source file (only once)
                if source_idx == 0:
                    with open(os.path.join(split_output_dir, f"{split_name}.source"), 'w') as f:
                        f.write(source_text + '\n')
                
                # Write augmentations
                augs = augmentation_mapping.get(source_idx, [])
                for aug_idx, aug_files in enumerate(aug_files):
                    if aug_idx < len(augs):
                        aug = augs[aug_idx]
                        summary = aug['summary']
                        summary_doc = nlp(str(summary))
                        entity_tokens = get_entity_tokens(summary_doc)
                        
                        # Write augmentation
                        aug_files['target'].write(summary + '\n')
                        aug_files['ne'].write(' '.join(map(str, entity_tokens)) + '\n')
                        aug_files['other'].write(f"{source_idx}\t{aug_idx}\n")
                    else:
                        # No augmentation available
                        aug_files['target'].write("[NO_AUG]\n")
                        aug_files['ne'].write("0\n")
                        aug_files['other'].write(f"{source_idx}\t-1\n")

            # Close all files
            for files in aug_files:
                for f in files.values():
                    f.close()

    return augmentation_mapping

def load_positive_summaries(dataset_dir):
    """Load positive summaries while preserving original indices."""
    pos_summaries = {}
    try:
        pos_dataset = load_from_disk(dataset_dir)
        print(f"\nLoading positive summaries from {dataset_dir}")
        print(f"Positive dataset size: {len(pos_dataset)}")
        
        # First check if we have original_idx in the dataset
        if 'original_idx' in pos_dataset.features:
            print("Using original_idx from dataset")
            for example in pos_dataset:
                if 'summary' in example and 'original_idx' in example:
                    pos_summaries[example['original_idx']] = example['summary']
        else:
            print("Using sequential indexing")
            for idx, example in enumerate(pos_dataset):
                if 'summary' in example:
                    pos_summaries[idx] = example['summary']
        
        # Print statistics
        print(f"Loaded {len(pos_summaries)} summaries")
        if pos_summaries:
            print(f"Index range: {min(pos_summaries.keys())} to {max(pos_summaries.keys())}")
            
        return pos_summaries
        
    except Exception as e:
        print(f"Error loading positive summaries: {e}")
        return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process HuggingFace datasets to create .source, .target, .ne, and .other files.")
    parser.add_argument('--data_dir', type=str, required=True, help="Base directory of the datasets.")
    parser.add_argument('--output_base_dir', type=str, required=True, help="Base directory for output files.")
    parser.add_argument('--aug_types', nargs='+', required=True, help="List of neg augmentation types to process")
    parser.add_argument('--splits_ratio', nargs=3, type=float, default=[0.9, 0.05, 0.05], help="Ratios for train, validation, and test splits.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for shuffling and splitting.")
    parser.add_argument('--tokenizer_names', nargs='+', default=['meta-llama/Llama-2-7b-hf', 'microsoft/phi-2', 'gpt2'], help="List of tokenizer names to process data for")
    parser.add_argument('--pos_aug_types', nargs='+', default=['original', 'back_translation'], help="Types of positive augmentations to process")
    parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', help="spaCy model name for entity extraction.")
    args = parser.parse_args()

    # Load all tokenizers
    tokenizers = {}
    for tokenizer_name in args.tokenizer_names:
        print(f"Loading tokenizer: {tokenizer_name}")
        tokenizers[tokenizer_name] = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    nlp = spacy.load(args.spacy_model)

    # Load positive summaries with new function
    pos_dataset_dir = os.path.join(args.data_dir, 'pos', 'original')
    if os.path.exists(pos_dataset_dir):
        pos_summaries = load_positive_summaries(pos_dataset_dir)
        
        if not pos_summaries:
            print("Error: Failed to load positive summaries")
            exit(1)
            
        # Add debug info about indices
        print("\nPositive summaries loaded:")
        print(f"Total summaries: {len(pos_summaries)}")
        print(f"Index range: {min(pos_summaries.keys())} to {max(pos_summaries.keys())}")
        print("6963", pos_summaries[6963])
        
        # Process each positive augmentation type with additional debug info
        for pos_type in args.pos_aug_types:
            dataset_dir = os.path.join(args.data_dir, 'pos', pos_type)

            # Debug the dataset before processing
            print(f"\nChecking {pos_type} dataset:")
            try:
                debug_dataset = load_from_disk(dataset_dir)
                print(f"Dataset size: {len(debug_dataset)}")
                print(f"Features: {debug_dataset.features}")
                print("Checking first few original_idx values:")
                for i, example in enumerate(debug_dataset[:5]):
                    orig_idx = example.get('original_idx', None)
                    print(f"Example {i}: original_idx = {orig_idx}")
                    if orig_idx is not None and orig_idx not in pos_summaries:
                        print(f"Warning: original_idx {orig_idx} not in pos_summaries")
            except Exception as e:
                print(f"Error checking dataset: {e}")
                
            output_dir = os.path.join(args.output_base_dir, 'pos', pos_type)
            process_dataset(f'pos/{pos_type}', dataset_dir, output_dir, 
                          tokenizers, nlp, pos_summaries, 
                          splits_ratio=args.splits_ratio, seed=args.seed)
                        
            # verify_data_alignment(
            #     source_file=f"{output_dir}/{tokenizer_name}/train/train.source",
            #     pos_file=f"{output_dir}/{tokenizer_name}/train/train.pos_target",
            #     f"{output_dir}/{tokenizer_name}/train/train.neg_target",
            #     f"{output_dir}/{tokenizer_name}/train/train.other",
            #     tokenizers[tokenizer_name]
            # )

            # verify_data_alignment(
            #     f"{output_dir}/train.source",
            #     f"{output_dir}/train.pos_target",
            #     f"{output_dir}/train.neg_target",
            #     f"{output_dir}/train.other",
            #     tokenizers[tokenizer_name]
            # )