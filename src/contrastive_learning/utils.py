import os
import sys
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
# from src.factuality_detector import FactualityDetector
import re
import torch
from transformers import pipeline

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

def calculate_original_factuality(data, factuality_detector, cache_dir='/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/cache', dataset_name='dataset'):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f'factuality_scores_{dataset_name}.json')
    
    # Check if cache file exists
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            factuality_scores = json.load(f)
        total_factuality_score = sum(factuality_scores)
        num_samples = len(factuality_scores)
        return total_factuality_score / num_samples
    else:
        # Compute factuality scores
        total_factuality_score = 0
        num_samples = 0
        factuality_scores = []
        for source, target in zip(data['source'], data['target']):
            _, _, factuality_score = factuality_detector.generate_score(
                [preprocess_text(source)], [preprocess_text(target)], summac_style=True
            )
            # Convert numpy.float32 to float
            factuality_score = float(factuality_score)
            factuality_scores.append(factuality_score)
            total_factuality_score += factuality_score
            num_samples += 1
        # Save scores to cache file
        with open(cache_file, 'w') as f:
            json.dump(factuality_scores, f)
        return total_factuality_score / num_samples

def calculate_original_fairness(data, fairness_evaluator):
    total_fairness_score = 0
    num_samples = 0
    for target in data['target']:
        if num_samples % 10 == 0:
            fairness_score = fairness_evaluator.calc_q_value(preprocess_text(target))
            total_fairness_score += fairness_score
        num_samples += 1
    return total_fairness_score / (num_samples // 10)

def read_files(directory, logger):
    data = {}
    for split in ['train']:
        try:
            data[split] = {
                'source': open(os.path.join(directory, f'{split}.source'), 'r').readlines(),
                'target': open(os.path.join(directory, f'{split}.target'), 'r').readlines(),
            }
            logger.info(f"Loaded data from {directory}")
        except:
            logger.info(f"Couldn't find {directory}")
    return data

def evaluate_toxicity(text):
    toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", device=0 if torch.cuda.is_available() else -1)
    results = toxicity_model(text, batch_size=1, truncation=True, max_length=512)
    return results[0]['score']

def calculate_original_toxicity(data, cache_dir='/gpfs/home/bsk18/factual-bias-mitigation/data/tldr', dataset_name='dataset'):
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Define cache file path
    cache_file = os.path.join(cache_dir, f'toxicity_scores_{dataset_name}.json')
    
    # Check if cache file exists
    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as f:
            toxicity_scores = json.load(f)
        total_toxicity_score = sum(toxicity_scores)
        num_samples = len(toxicity_scores)
        return total_toxicity_score / num_samples
    else:
        # Compute toxicity scores
        total_toxicity_score = 0
        num_samples = 0
        toxicity_scores = []
        for target in data['target']:
            toxicity_score = evaluate_toxicity(target)
            toxicity_score = float(toxicity_score)
            toxicity_scores.append(toxicity_score)
            total_toxicity_score += toxicity_score
            num_samples += 1
        # Save scores to cache file
        with open(cache_file, 'w') as f:
            json.dump(toxicity_scores, f)
        return total_toxicity_score / num_samples

def generate_dataset_name(data_type, data_options):
    # data_type: 'pos' or 'neg'
    # data_options: list of data options used
    return f"{data_type}_{'_'.join(data_options)}"