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
            'neg_source_ids': torch.cat([e['input_ids'] for e in neg_source_encodings]),
            'neg_source_attention_mask': torch.cat([e['attention_mask'] for e in neg_source_encodings]),
            'neg_target_ids': torch.cat([e['input_ids'] for e in neg_target_encodings]),
            'neg_target_attention_mask': torch.cat([e['attention_mask'] for e in neg_target_encodings]),
            'neg_combined_ids': torch.cat([e['input_ids'] for e in neg_combined_encodings]),
            'neg_combined_attention_mask': torch.cat([e['attention_mask'] for e in neg_combined_encodings]),
        }
