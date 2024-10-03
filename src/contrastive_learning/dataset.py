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

        # Tokenize without padding to get true lengths
        pos_source_encoding = self.tokenizer(
            pos_source, truncation=True, max_length=self.max_length, return_tensors='pt'
        )
        pos_target_encoding = self.tokenizer(
            pos_target, truncation=True, max_length=self.max_length, return_tensors='pt'
        )

        source_length = pos_source_encoding['input_ids'].size(1)
        target_length = pos_target_encoding['input_ids'].size(1)

        # Concatenate input_ids and attention_mask
        input_ids = torch.cat(
            [pos_source_encoding['input_ids'], pos_target_encoding['input_ids']], dim=1
        )
        attention_mask = torch.cat(
            [pos_source_encoding['attention_mask'], pos_target_encoding['attention_mask']], dim=1
        )

        # Create labels
        labels = input_ids.clone()
        labels[:, :source_length] = -100  # Ignore source tokens

        # Truncate or pad to max_length
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]
        else:
            padding_length = self.max_length - input_ids.size(1)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, padding_length), value=self.tokenizer.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, padding_length), value=0
            )
            labels = torch.nn.functional.pad(
                labels, (0, padding_length), value=-100
            )
        # After creating labels
        valid_labels = labels[labels != -100]

        if valid_labels.numel() == 0:
            # All labels are -100, handle this case
            print(f"Sample at index {idx} has an empty target after tokenization. Skipping.")
            return self.__getitem__(random.randint(0, len(self) - 1))  # Fetch another sample

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
            'labels': labels,
            'input_ids': input_ids.squeeze(0),
            'attention_mask': attention_mask.squeeze(0),
            'pos_source_ids': pos_source_encoding['input_ids'].squeeze(),
            'pos_source_attention_mask': pos_source_encoding['attention_mask'].squeeze(),
            'pos_target_ids': pos_target_encoding['input_ids'].squeeze(),
            'pos_target_attention_mask': pos_target_encoding['attention_mask'].squeeze(),
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

        combined_encoding = self.tokenizer(
                source,
                target,
                truncation='only_first',  # Truncate only the source text
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt',
                add_special_tokens=True  # Ensure special tokens are added if necessary
            )

        combined_ids = combined_encoding['input_ids'].squeeze()
        attention_mask = combined_encoding['attention_mask'].squeeze()

        # Create labels
        labels = combined_ids.clone()

        # Tokenize source separately to get the length
        source_encoding = self.tokenizer(
            source,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        source_length = source_encoding['input_ids'].size(1)

        # Adjust source_length if combined length exceeds max_length
        total_length = combined_ids.size(0)
        target_length = total_length - source_length
        if total_length > self.max_length:
            source_length = total_length - target_length

        # Set labels to -100 for source tokens
        labels[:source_length] = -100
        # Tokenize the source separately to get pos_source_ids and pos_source_attention_mask for generation
        pos_source_encoding = self.tokenizer(
            source,
            truncation=True,
            padding='max_length',
            max_length=source_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        pos_source_ids = pos_source_encoding['input_ids'].squeeze(0)  # Squeeze batch dimension
        pos_source_attention_mask = pos_source_encoding['attention_mask'].squeeze(0)  # Squeeze batch dimension

        return {
            'is_positive': is_positive,
            'full_prompt': source,
            'combined_ids': combined_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'pos_source_ids': pos_source_ids,  # For generation
            'pos_source_attention_mask': pos_source_attention_mask,  # For generation
        }