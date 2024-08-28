import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
# from src.factuality_detector import FactualityDetector
import re
import torch

def combine_data(data_list):
    combined = {split: {'source': [], 'target': []} for split in ['train', 'validation', 'test']}
    for data in data_list:
        for split in ['train', 'validation', 'test']:
            combined[split]['source'].extend(data[split]['source'])
            combined[split]['target'].extend(data[split]['target'])
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

def calculate_original_factuality(data, factuality_detector):
    total_factuality_score = 0
    num_samples = 0
    for source, target in zip(data['source'], data['target']):
        _, _, factuality_score = factuality_detector.generate_score([preprocess_text(source)], [preprocess_text(target)], summac_style=True)
        total_factuality_score += factuality_score
        num_samples += 1
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
    for split in ['train', 'validation', 'test']:
        data[split] = {
            'source': open(os.path.join(directory, f'{split}.source'), 'r').readlines(),
            'target': open(os.path.join(directory, f'{split}.target'), 'r').readlines(),
        }
    logger.info(f"Loaded data from {directory}")
    return data
