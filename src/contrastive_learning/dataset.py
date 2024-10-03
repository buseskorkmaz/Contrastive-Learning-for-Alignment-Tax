import random
import torch
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, pos_data, neg_data, tokenizer, max_length, neg_samples_per_pos=4):
        self.pos_data = pos_data
        self.neg_data = neg_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.neg_samples_per_pos = neg_samples_per_pos
        self.full_prompts = pos_data['source']

    def __len__(self):
        return len(self.pos_data['source'])

    def __getitem__(self, idx):
        pos_source = self.pos_data['source'][idx].strip()
        pos_target = self.pos_data['target'][idx].strip()

        # Randomly sample negative examples
        neg_indices = random.sample(range(len(self.neg_data['source'])), self.neg_samples_per_pos)
        neg_sources = [self.neg_data['source'][i].strip() for i in neg_indices]
        neg_targets = [self.neg_data['target'][i].strip() for i in neg_indices]

        # Encode positive example
        pos_source_encoding = self.tokenizer(pos_source, truncation=True, padding='max_length', 
                                            max_length=self.max_length, return_tensors='pt')
        pos_target_encoding = self.tokenizer(pos_target, truncation=True, padding='max_length', 
                                            max_length=self.max_length, return_tensors='pt')
        pos_combined_encoding = self.tokenizer(pos_source, pos_target, truncation=True, padding='max_length', 
                                            max_length=self.max_length, return_tensors='pt')

        # Encode negative examples
        neg_source_encodings = [self.tokenizer(ns, truncation=True, padding='max_length', 
                                               max_length=self.max_length, return_tensors='pt') for ns in neg_sources]
        neg_target_encodings = [self.tokenizer(nt, truncation=True, padding='max_length', 
                                               max_length=self.max_length, return_tensors='pt') for nt in neg_targets]
        neg_combined_encodings = [self.tokenizer(ns, nt, truncation=True, padding='max_length', 
                                                 max_length=self.max_length, return_tensors='pt') 
                                  for ns, nt in zip(neg_sources, neg_targets)]

        return {
            'full_prompt': self.full_prompts[idx],
            'pos_source_ids': pos_source_encoding['input_ids'].squeeze(),
            'pos_source_attention_mask': pos_source_encoding['attention_mask'].squeeze(),
            'pos_target_ids': pos_target_encoding['input_ids'].squeeze(),
            'pos_target_attention_mask': pos_target_encoding['attention_mask'].squeeze(),
            'pos_combined_ids': pos_combined_encoding['input_ids'].squeeze(),
            'pos_combined_attention_mask': pos_combined_encoding['attention_mask'].squeeze(),
            'neg_source_ids': torch.stack([e['input_ids'].squeeze() for e in neg_source_encodings]),
            'neg_source_attention_mask': torch.stack([e['attention_mask'].squeeze() for e in neg_source_encodings]),
            'neg_target_ids': torch.stack([e['input_ids'].squeeze() for e in neg_target_encodings]),
            'neg_target_attention_mask': torch.stack([e['attention_mask'].squeeze() for e in neg_target_encodings]),
            'neg_combined_ids': torch.stack([e['input_ids'].squeeze() for e in neg_combined_encodings]),
            'neg_combined_attention_mask': torch.stack([e['attention_mask'].squeeze() for e in neg_combined_encodings]),
        }

class EvaluationDataset(Dataset):
    def __init__(self, pos_data_path, neg_data_path, tokenizer, max_length, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        random.seed(seed)
        torch.manual_seed(seed)

        self.pos_data = self.read_files(pos_data_path)
        self.neg_data = self.read_files(neg_data_path)

        self.pos_length = len(self.pos_data['source'])
        self.neg_length = len(self.neg_data['source'])
        self.total_length = self.pos_length + self.neg_length

    def read_files(self, data_path):
        if "test" in data_path:
            with open(f"{data_path}/test.source", 'r') as f:
                source = f.readlines()
            with open(f"{data_path}/test.target", 'r') as f:
                target = f.readlines()
        else:
            with open(f"{data_path}/validation.source", 'r') as f:
                source = f.readlines()
            with open(f"{data_path}/validation.target", 'r') as f:
                target = f.readlines()
        return {'source': source, 'target': target}

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < self.pos_length:
            is_positive = True
            source = self.pos_data['source'][idx].strip()
            target = self.pos_data['target'][idx].strip()
        else:
            is_positive = False
            idx = idx - self.pos_length
            source = self.neg_data['source'][idx].strip()
            target = self.neg_data['target'][idx].strip()

        source_encoding = self.tokenizer(source, truncation=True, padding='max_length', 
                                         max_length=self.max_length, return_tensors='pt')
        target_encoding = self.tokenizer(target, truncation=True, padding='max_length', 
                                         max_length=self.max_length, return_tensors='pt')
        combined_encoding = self.tokenizer(source, target, truncation=True, padding='max_length', 
                                           max_length=self.max_length, return_tensors='pt')

        return {
            'is_positive': is_positive,
            'full_prompt': source,
            'source_ids': source_encoding['input_ids'].squeeze(),
            'source_attention_mask': source_encoding['attention_mask'].squeeze(),
            'target_ids': target_encoding['input_ids'].squeeze(),
            'target_attention_mask': target_encoding['attention_mask'].squeeze(),
            'combined_ids': combined_encoding['input_ids'].squeeze(),
            'combined_attention_mask': combined_encoding['attention_mask'].squeeze(),
        }