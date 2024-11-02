import os
import json
from transformers import GPT2Tokenizer
from collections import defaultdict
from tqdm import tqdm
import argparse
import re
from difflib import SequenceMatcher
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize the model globally
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_repetition_rate(text, n=3):
    words = text.split()
    if len(words) < n:
        return 0
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    max_count = max(ngram_counts.values())
    total_ngrams = len(ngrams)
    repetition_rate = max_count / total_ngrams
    return repetition_rate


def is_similar_source_target(src_text, tgt_text):
    src_clean = remove_summary_prefix(src_text)
    tgt_clean = remove_summary_prefix(tgt_text)
    # Remove punctuation and convert to lowercase
    src_clean = re.sub(r'[^\w\s]', '', src_clean.lower())
    tgt_clean = re.sub(r'[^\w\s]', '', tgt_clean.lower())
    ratio = SequenceMatcher(None, src_clean, tgt_clean).ratio()
    return ratio  # Return the similarity ratio


def remove_summary_prefix(text):
    return text.replace("Summary:", "").strip()

def is_repetitive_text(text, similarity_threshold=0.8, repetition_ratio_threshold=0.5):
    # Preprocess the text
    text_clean = re.sub(r'\s+', ' ', text.strip())
    
    # Split text into sentences
    sentences = sent_tokenize(text_clean)
    
    # Remove very short sentences (less than 3 words)
    sentences = [s for s in sentences if len(s.split()) >= 3]
    
    if len(sentences) < 2:
        return False, 0  # Not enough sentences to compare, repetition ratio is 0
    
    # Compute sentence embeddings
    embeddings = sentence_model.encode(sentences, convert_to_tensor=True)
    
    # Compute pairwise cosine similarities
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)
    
    # Count the number of pairs with high similarity
    num_sentences = len(sentences)
    num_pairs = num_sentences * (num_sentences - 1) / 2
    high_sim_pairs = 0
    
    for i in range(num_sentences):
        for j in range(i+1, num_sentences):
            if cosine_scores[i][j] >= similarity_threshold:
                high_sim_pairs += 1
    
    # Calculate the repetition ratio
    repetition_ratio = high_sim_pairs / num_pairs if num_pairs > 0 else 0
    is_repetitive = repetition_ratio >= repetition_ratio_threshold
    
    return is_repetitive, repetition_ratio  # Return both the flag and the repetition ratio



def read_and_clean_files(data_dir, split, tokenizer, output_dir, clean=True):
    similarity_threshold = 0.9  # Threshold for source-target similarity
    repetition_similarity_threshold = 0.8  # Threshold for sentence similarity in target text
    repetition_ratio_threshold = 0.3  # Threshold for repetition ratio in target text

    source_file = os.path.join(data_dir, f'{split}.source')
    target_file = os.path.join(data_dir, f'{split}.target')
    
    if not os.path.exists(source_file) or not os.path.exists(target_file):
        print(f"Source or target file not found in {data_dir} for split {split}")
        return None, None
    
    with open(source_file, 'r', encoding='utf-8') as f_src, open(target_file, 'r', encoding='utf-8') as f_tgt:
        source_texts = [line.strip() for line in f_src]
        target_texts = [line.strip() for line in f_tgt]

    if clean:
        cleaned_source_texts = []
        cleaned_target_texts = []
        removed_samples_log = os.path.join(output_dir, f'removed_samples-{split}.txt')
        sample_metrics_log = os.path.join(output_dir, f'sample_metrics_log-{split}.txt')

        with open(sample_metrics_log, 'w', encoding='utf-8') as f_metrics_log, \
             open(removed_samples_log, 'w', encoding='utf-8') as f_log:
            for src_text, tgt_text in tqdm(zip(source_texts, target_texts), total=len(source_texts), desc=f"Cleaning data for {split} in {data_dir}"):
                src_clean = remove_summary_prefix(src_text)
                tgt_clean = remove_summary_prefix(tgt_text)

                # Compute similarity between source and target
                similarity_ratio = is_similar_source_target(src_clean, tgt_clean)
                is_similar = similarity_ratio >= similarity_threshold

                # Compute repetition in target text
                is_repetitive, repetition_ratio = is_repetitive_text(tgt_clean, similarity_threshold=repetition_similarity_threshold, repetition_ratio_threshold=repetition_ratio_threshold)

                # Log the sample and metrics
                f_metrics_log.write(f"Source: {src_text}\n")
                f_metrics_log.write(f"Target: {tgt_text}\n")
                f_metrics_log.write(f"Similarity Ratio: {similarity_ratio:.4f}\n")
                f_metrics_log.write(f"Is Similar: {is_similar}\n")
                f_metrics_log.write(f"Repetition Ratio: {repetition_ratio:.4f}\n")
                f_metrics_log.write(f"Is Repetitive: {is_repetitive}\n")

                if is_similar:
                    f_metrics_log.write("Status: Removed due to high similarity between source and target.\n")
                    f_metrics_log.write("-" * 50 + "\n")
                    f_log.write(f"Similar source and target removed:\nSource: {src_text}\nTarget: {tgt_text}\n\n")
                    continue
                if is_repetitive:
                    f_metrics_log.write("Status: Removed due to repetition in target text.\n")
                    f_metrics_log.write("-" * 50 + "\n")
                    f_log.write(f"Repetitive target text removed:\nSource: {src_text}\nTarget: {tgt_text}\n\n")
                    continue

                # If the sample is kept
                f_metrics_log.write("Status: Kept.\n")
                f_metrics_log.write("-" * 50 + "\n")

                cleaned_source_texts.append(src_text)
                cleaned_target_texts.append(tgt_text)

        if not cleaned_source_texts or not cleaned_target_texts:
            print(f"No data left after cleaning for {split} in {data_dir}")
            return None, None
    else:
        # No cleaning applied
        cleaned_source_texts = source_texts
        cleaned_target_texts = target_texts

    # Save data into appropriate directory
    if clean:
        data_subdir = 'cleaned_data'
    else:
        data_subdir = 'uncleaned_data'

    data_dir_final = os.path.join(output_dir, data_subdir, os.path.basename(data_dir))
    os.makedirs(data_dir_final, exist_ok=True)

    with open(os.path.join(data_dir_final, f'{split}.source'), 'w', encoding='utf-8') as f_src, \
         open(os.path.join(data_dir_final, f'{split}.target'), 'w', encoding='utf-8') as f_tgt:
        for src_text, tgt_text in zip(cleaned_source_texts, cleaned_target_texts):
            f_src.write(src_text + '\n')
            f_tgt.write(tgt_text + '\n')

    return cleaned_source_texts, cleaned_target_texts



def create_index_mappings(
    tokenizer,
    source_texts,
    target_texts,
    pos_data_paths,
    neg_data_paths,
    split,
    output_dir
):
    # Build a mapping from source text to its indices
    source_to_indices = defaultdict(list)
    for idx, src_text in enumerate(source_texts):
        source_to_indices[src_text].append(idx)
    
    # Now, for each source text, collect indices of positive and negative targets
    pos_mappings = []
    neg_mappings = []
    
    # Update data paths to use cleaned data for positives and uncleaned data for negatives
    cleaned_pos_data_paths = [os.path.join(output_dir, 'cleaned_data', os.path.basename(p)) for p in pos_data_paths]
    uncleaned_neg_data_paths = [os.path.join(output_dir, 'uncleaned_data', os.path.basename(p)) for p in neg_data_paths]
    
    # Load all positive target texts (cleaned)
    all_pos_targets = []
    for pos_path in cleaned_pos_data_paths:
        _, pos_targets = read_and_clean_files(pos_path, split, tokenizer, output_dir, clean=False)  # Avoid double cleaning
        if pos_targets:
            all_pos_targets.extend(pos_targets)
    
    # Load all negative target texts (uncleaned)
    all_neg_targets = []
    for neg_path in uncleaned_neg_data_paths:
        _, neg_targets = read_and_clean_files(neg_path, split, tokenizer, output_dir, clean=False)
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split in args.splits:
        print(f"Processing split: {split}")
        
        # For training, use specified data options
        if split == 'train':
            pos_data_options = args.pos_data_options
            neg_data_options = args.neg_data_options
        else:
            # For validation and test, only use 'original' data
            pos_data_options = ['original']
            neg_data_options = ['original']
        
        # Prepare data paths
        pos_data_paths = [os.path.join(args.data_dir, f'pos_{opt}') if opt != 'original' else os.path.join(args.data_dir, 'pos') for opt in pos_data_options]
        neg_data_paths = [os.path.join(args.data_dir, f'neg_{opt}') if opt != 'original' else os.path.join(args.data_dir, 'neg') for opt in neg_data_options]
        
        source_texts = []
        target_texts = []
        for pos_path in pos_data_paths:
            src_texts, tgt_texts = read_and_clean_files(pos_path, split, tokenizer, args.output_dir, clean=True)
            if src_texts and tgt_texts:
                source_texts.extend(src_texts)
                target_texts.extend(tgt_texts)
        
        if not source_texts or not target_texts:
            print(f"No positive data found after cleaning for split {split}")
            continue
        
        # Process negative data without cleaning
        for neg_path in neg_data_paths:
            read_and_clean_files(neg_path, split, tokenizer, args.output_dir, clean=False)
            # Negative data is handled in create_index_mappings
        
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