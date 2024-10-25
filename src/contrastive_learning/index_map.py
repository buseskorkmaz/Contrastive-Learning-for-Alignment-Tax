import os
import json
from transformers import GPT2Tokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse


def read_and_tokenize_files(data_dir, split, tokenizer):
    source_file = os.path.join(data_dir, f'{split}.source')
    target_file = os.path.join(data_dir, f'{split}.target')
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"Source or target file not found in {data_dir} for split {split}")
        return None, None
    
    with open(source_file, 'r', encoding='utf-8') as f_src, open(target_file, 'r', encoding='utf-8') as f_tgt:
        source_texts = [line.strip() for line in f_src]
        target_texts = [line.strip() for line in f_tgt]
    
    # Tokenize texts
    source_encodings = tokenizer(
        source_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    target_encodings = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    return source_texts, target_texts


def create_index_mappings(
    tokenizer,
    source_texts,
    target_texts,
    pos_data_paths,
    neg_data_paths,
    split,
    output_dir
):
    # For each sample in source_texts, we need to find the indices of its positive and negative targets
    # Since we are dealing with augmented data, we'll assume that the source texts are repeated for each augmentation
    # We'll create mappings based on matching source texts

    # Build a mapping from source text to its indices
    source_to_indices = defaultdict(list)
    for idx, src_text in enumerate(source_texts):
        source_to_indices[src_text].append(idx)
    
    # Now, for each source text, collect indices of positive and negative targets
    pos_mappings = []
    neg_mappings = []
    
    # Load all positive and negative target texts
    all_pos_targets = []
    for pos_path in pos_data_paths:
        _, pos_targets = read_and_tokenize_files(pos_path, split, tokenizer)
        if pos_targets:
            all_pos_targets.extend(pos_targets)
    
    all_neg_targets = []
    for neg_path in neg_data_paths:
        _, neg_targets = read_and_tokenize_files(neg_path, split, tokenizer)
        if neg_targets:
            all_neg_targets.extend(neg_targets)
    
    # Build index mappings
    for idx, src_text in enumerate(tqdm(source_texts, desc=f'Creating index mappings for {split}')):
        # Positive indices
        pos_indices = []
        for i, tgt_text in enumerate(all_pos_targets):
            if tgt_text == target_texts[idx]:
                pos_indices.append(i)
        if not pos_indices:
            # If no positive targets found, we can default to the current index
            pos_indices.append(idx)
        pos_mappings.append(pos_indices)
        
        # Negative indices
        neg_indices = []
        for i in range(len(all_neg_targets)):
            neg_indices.append(i)
        neg_mappings.append(neg_indices)
    
    # Save mappings to files
    pos_index_file = os.path.join(output_dir, f'{split}.positive.index')
    neg_index_file = os.path.join(output_dir, f'{split}.negative.index')
    
    with open(pos_index_file, 'w') as f_pos, open(neg_index_file, 'w') as f_neg:
        for pos_indices in pos_mappings:
            f_pos.write(' '.join(map(str, pos_indices)) + '\n')
        for neg_indices in neg_mappings:
            f_neg.write(' '.join(map(str, neg_indices)) + '\n')
    
    print(f"Saved positive index mappings to {pos_index_file}")
    print(f"Saved negative index mappings to {neg_index_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Base data directory')
    parser.add_argument('--pos_data_options', nargs='+', default=['original'], help='Positive data options')
    parser.add_argument('--neg_data_options', nargs='+', default=['original'], help='Negative data options')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'], help='Data splits to process')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save index files')
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare data paths
    pos_data_paths = [os.path.join(args.data_dir, f'pos_{opt}') if opt != 'original' else os.path.join(args.data_dir, 'pos') for opt in args.pos_data_options]
    neg_data_paths = [os.path.join(args.data_dir, f'neg_{opt}') if opt != 'original' else os.path.join(args.data_dir, 'neg') for opt in args.neg_data_options]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split in args.splits:
        print(f"Processing split: {split}")
        if split == 'train':
            
            # Read and tokenize source and target files
            source_texts = []
            target_texts = []
            for pos_path in pos_data_paths:
                src_texts, tgt_texts = read_and_tokenize_files(pos_path, split, tokenizer)
                if src_texts and tgt_texts:
                    source_texts.extend(src_texts)
                    target_texts.extend(tgt_texts)
        else:
            pos_data_paths = [f'/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/{split}/pos']
            neg_data_paths = [f'/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/{split}/neg']
            # Read and tokenize source and target files
            source_texts = []
            target_texts = []
            for pos_path in pos_data_paths:
                src_texts, tgt_texts = read_and_tokenize_files(pos_path, split, tokenizer)
                if src_texts and tgt_texts:
                    source_texts.extend(src_texts)
                    target_texts.extend(tgt_texts)
            
        if not source_texts or not target_texts:
            print(f"No data found for split {split}")
            continue
    
        # Create index mappings
        create_index_mappings(
            tokenizer,
            source_texts,
            target_texts,
            pos_data_paths,
            neg_data_paths,
            split,
            args.output_dir
        )

if __name__ == "__main__":
    main()