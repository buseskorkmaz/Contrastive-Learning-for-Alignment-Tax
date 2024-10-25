import os
import sys
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
# from src.factuality_detector import FactualityDetector
import re
import torch
from transformers import pipeline
from tqdm import tqdm

def combine_data(data_list):
    combined = {split: {'source': [], 'target': []} for split in ['train', 'validation', 'test']}
    for data in data_list:
        try:
            for split in ['train', 'validation', 'test']:
                combined[split]['source'].extend(data[split]['source'])
                combined[split]['target'].extend(data[split]['target'])
        except:
            print(f"skipping {split}, couldn't find")
    return combined

def print_gpu_memory(epoch):
    if torch.cuda.is_available():
        print(f"Epoch {epoch}: GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

def remove_unexpected_keys(state_dict):
    return {k: v for k, v in state_dict.items() if not (k.endswith('.attn.bias') or k.endswith('.attn.masked_bias'))}

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string inputs
    pattern = r"Based on the original.*$"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip().lower()

def calculate_original_factuality(dataset, factuality_detector, cache_dir='/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/cache', dataset_name='dataset'):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file paths
    pos_cache_file = os.path.join(cache_dir, f'factuality_scores_{dataset_name}_positive.json')
    neg_cache_file = os.path.join(cache_dir, f'factuality_scores_{dataset_name}_negative.json')
    
    # Check if cache files exist
    pos_scores_exist = os.path.isfile(pos_cache_file)
    neg_scores_exist = os.path.isfile(neg_cache_file)
    
    pos_factuality_scores = []
    neg_factuality_scores = []

    if pos_scores_exist and neg_scores_exist:
        with open(pos_cache_file, 'r') as f:
            pos_factuality_scores = json.load(f)
        with open(neg_cache_file, 'r') as f:
            neg_factuality_scores = json.load(f)
    else:
        for sample in tqdm(dataset, desc="Calculating Original Factuality"):
            src_text = sample['src_text']
            
            # Positive targets
            for target in sample['pos_texts']:
                _, _, factuality_score = factuality_detector.generate_score(
                    [preprocess_text(src_text)], [preprocess_text(target)], summac_style=True
                )
                factuality_score = float(factuality_score)
                pos_factuality_scores.append(factuality_score)
            
            # Negative targets
            for target in sample['neg_texts']:
                _, _, factuality_score = factuality_detector.generate_score(
                    [preprocess_text(src_text)], [preprocess_text(target)], summac_style=True
                )
                factuality_score = float(factuality_score)
                neg_factuality_scores.append(factuality_score)

        # Save scores to cache files
        with open(pos_cache_file, 'w') as f:
            json.dump(pos_factuality_scores, f)
        with open(neg_cache_file, 'w') as f:
            json.dump(neg_factuality_scores, f)
    
    # Calculate average factuality scores
    avg_pos_factuality = sum(pos_factuality_scores) / len(pos_factuality_scores) if pos_factuality_scores else 0
    avg_neg_factuality = sum(neg_factuality_scores) / len(neg_factuality_scores) if neg_factuality_scores else 0

    return avg_pos_factuality, avg_neg_factuality

def calculate_original_fairness(data, fairness_evaluator):
    total_fairness_score = 0
    num_samples = 0
    for target in data['target']:
        if num_samples % 10 == 0:
            fairness_score = fairness_evaluator.calc_q_value(preprocess_text(target))
            total_fairness_score += fairness_score
        num_samples += 1
    return total_fairness_score / (num_samples // 10)

def read_files(directory, logger, splits=None):
    data = {}
    if not splits:
        splits = ['train']

    for split in splits:
        try:
            data[split] = {
                'source': open(os.path.join(directory, f'{split}.source'), 'r').readlines(),
                'target': open(os.path.join(directory, f'{split}.target'), 'r').readlines(),
            }
            logger.info(f"Loaded data from {directory}")
        except:
            logger.info(f"Couldn't find {directory}")
    return data

def load_indices(index_file):
    indices = []
    with open(index_file, 'r') as f:
        for line in f:
            idx_list = [int(idx) for idx in line.strip().split()]
            indices.append(idx_list)
    return indices

def evaluate_toxicity(text):
    toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", device=0 if torch.cuda.is_available() else -1)
    results = toxicity_model(text, batch_size=1, truncation=True, max_length=512)
    return results[0]['score']

def calculate_original_toxicity(dataset, cache_dir='/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/cache', dataset_name='dataset'):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    # Define cache file paths
    pos_cache_file = os.path.join(cache_dir, f'toxicity_scores_{dataset_name}_positive.json')
    neg_cache_file = os.path.join(cache_dir, f'toxicity_scores_{dataset_name}_negative.json')
    
    # Check if cache files exist
    pos_scores_exist = os.path.isfile(pos_cache_file)
    neg_scores_exist = os.path.isfile(neg_cache_file)
    
    pos_toxicity_scores = []
    neg_toxicity_scores = []

    if pos_scores_exist and neg_scores_exist:
        with open(pos_cache_file, 'r') as f:
            pos_toxicity_scores = json.load(f)
        with open(neg_cache_file, 'r') as f:
            neg_toxicity_scores = json.load(f)
    else:
        toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", device=0 if torch.cuda.is_available() else -1)
        for sample in tqdm(dataset, desc="Calculating Original Toxicity"):
            # Positive targets
            for target in sample['pos_texts']:
                toxicity_score = toxicity_model([target], batch_size=1, truncation=True, max_length=512)[0]['score'][0]
                toxicity_score = float(toxicity_score)
                pos_toxicity_scores.append(toxicity_score)
            
            # Negative targets
            for target in sample['neg_texts']:
                toxicity_score = toxicity_model([target], batch_size=1, truncation=True, max_length=512)[0]['score'][0]
                toxicity_score = float(toxicity_score)
                neg_toxicity_scores.append(toxicity_score)

        # Save scores to cache files
        with open(pos_cache_file, 'w') as f:
            json.dump(pos_toxicity_scores, f)
        with open(neg_cache_file, 'w') as f:
            json.dump(neg_toxicity_scores, f)
    
    # Calculate average toxicity scores
    avg_pos_toxicity = sum(pos_toxicity_scores) / len(pos_toxicity_scores) if pos_toxicity_scores else 0
    avg_neg_toxicity = sum(neg_toxicity_scores) / len(neg_toxicity_scores) if neg_toxicity_scores else 0

    return avg_pos_toxicity, avg_neg_toxicity


def generate_dataset_name(data_type, data_options):
    # data_type: 'pos' or 'neg'
    # data_options: list of data options used
    return f"{data_type}_{'_'.join(data_options)}"