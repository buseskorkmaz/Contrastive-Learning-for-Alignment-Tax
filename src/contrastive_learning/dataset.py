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
        
        # Set random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)

        # Load positive and negative data
        self.pos_data = self.read_files(pos_data_path)
        self.neg_data = self.read_files(neg_data_path)

        # Ensure positive and negative data have the same length
        assert len(self.pos_data['source']) == len(self.pos_data['target']), "Source and target data must have the same length"

        # # Combine and shuffle the data
        # combined_data = list(zip(self.pos_data['source'], self.pos_data['target'], 
        #                          self.neg_data['source'], self.neg_data['target']))
        # random.shuffle(combined_data)

        # # Unpack the shuffled data
        # self.pos_source, self.pos_target, self.neg_source, self.neg_target = zip(*combined_data)

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
        return len(self.pos_data['source']) + len(self.neg_data['source'])

    def __getitem__(self, idx):
        # print(type(idx), idx)
        # print(self.pos_data)
        # print(self.neg_data)
        # print(self.pos_data.keys())
        pos_source = self.pos_data['source'][idx].strip()
        pos_target = self.pos_data['target'][idx].strip()
        neg_source = self.neg_data['source'][idx].strip()
        neg_target = self.neg_data['target'][idx].strip()

        # Encode positive samples
        pos_source_encoding = self.tokenizer(pos_source, truncation=True, padding='max_length', 
                                             max_length=self.max_length, return_tensors='pt')
        pos_target_encoding = self.tokenizer(pos_target, truncation=True, padding='max_length', 
                                             max_length=self.max_length, return_tensors='pt')
        pos_combined_encoding = self.tokenizer(pos_source, pos_target, truncation=True, padding='max_length', 
                                               max_length=self.max_length, return_tensors='pt')

        # Encode negative samples
        neg_source_encoding = self.tokenizer(neg_source, truncation=True, padding='max_length', 
                                             max_length=self.max_length, return_tensors='pt')
        neg_target_encoding = self.tokenizer(neg_target, truncation=True, padding='max_length', 
                                             max_length=self.max_length, return_tensors='pt')
        neg_combined_encoding = self.tokenizer(neg_source, neg_target, truncation=True, padding='max_length', 
                                               max_length=self.max_length, return_tensors='pt')

        return {
            'pos_full_prompt': pos_source,
            'neg_full_prompt': neg_source,
            'pos_source_ids': pos_source_encoding['input_ids'].squeeze(),
            'pos_source_attention_mask': pos_source_encoding['attention_mask'].squeeze(),
            'pos_target_ids': pos_target_encoding['input_ids'].squeeze(),
            'pos_target_attention_mask': pos_target_encoding['attention_mask'].squeeze(),
            'pos_combined_ids': pos_combined_encoding['input_ids'].squeeze(),
            'pos_combined_attention_mask': pos_combined_encoding['attention_mask'].squeeze(),
            'neg_source_ids': neg_source_encoding['input_ids'].squeeze(),
            'neg_source_attention_mask': neg_source_encoding['attention_mask'].squeeze(),
            'neg_target_ids': neg_target_encoding['input_ids'].squeeze(),
            'neg_target_attention_mask': neg_target_encoding['attention_mask'].squeeze(),
            'neg_combined_ids': neg_combined_encoding['input_ids'].squeeze(),
            'neg_combined_attention_mask': neg_combined_encoding['attention_mask'].squeeze(),
        }