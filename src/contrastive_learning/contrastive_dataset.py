import torch
from torch.utils.data import Dataset
import numpy as np

class ContrastiveTranslationDataset(Dataset):
    def __init__(
        self,
        src_texts,
        pos_texts,
        neg_texts,
        pos_indices,
        neg_indices,
        tokenizer,
        max_length=512,
        max_neg_samples=5,
        cl_seed=0
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neg_samples = max_neg_samples
        self.cl_seed = cl_seed

        self.src_texts = src_texts  # List of source texts
        self.pos_texts = pos_texts  # List of positive target texts
        self.neg_texts = neg_texts  # List of negative target texts

        self.pos_indices = pos_indices  # List of lists of indices
        self.neg_indices = neg_indices  # List of lists of indices

        # Tokenize positive and negative target texts once
        self.pos_tokenized_texts = [
            self.tokenizer(
                text.strip(),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for text in self.pos_texts
        ]

        self.neg_tokenized_texts = [
            self.tokenizer(
                text.strip(),
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for text in self.neg_texts
        ]
        print(f"len(self.src_texts): {len(self.src_texts)}")
        print(f"len(self.pos_indices): {len(self.pos_indices)}")
        print(f"len(self.neg_indices): {len(self.neg_indices)}")

        # Initialize random number generator for negative sampling
        self.rng = np.random.default_rng(seed=cl_seed)
        self.neg_shuffle = [self.rng.permutation(mapping) for mapping in self.neg_indices]

        assert len(self.src_texts) == len(self.pos_indices) == len(self.neg_indices), "Data size mismatch"

    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        # Tokenize source text
        src_text = self.src_texts[idx].strip()
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids = src_encoding['input_ids'].squeeze()
        src_attention_mask = src_encoding['attention_mask'].squeeze()

        # Positive targets
        pos_indices = self.pos_indices[idx]
        pos_texts = [self.pos_texts[i].strip() for i in pos_indices]
        pos_input_ids_list = [self.pos_tokenized_texts[i]['input_ids'].squeeze() for i in pos_indices]
        pos_ne_masks = [torch.ones_like(ids) for ids in pos_input_ids_list]
        for mask in pos_ne_masks:
            mask[-1] = 0  # Adjust NE mask as needed

        # Negative targets
        neg_indices = self.neg_shuffle[idx]
        neg_texts = [self.neg_texts[i].strip() for i in neg_indices[:self.max_neg_samples]]
        neg_input_ids_list = [self.neg_tokenized_texts[i]['input_ids'].squeeze() for i in neg_indices[:self.max_neg_samples]]
        neg_ne_masks = [torch.ones_like(ids) for ids in neg_input_ids_list]
        for mask in neg_ne_masks:
            mask[-1] = 0  # Adjust NE mask as needed

        sample = {
            'id': idx,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'pos_input_ids_list': pos_input_ids_list,
            'neg_input_ids_list': neg_input_ids_list,
            'pos_ne_masks': pos_ne_masks,
            'neg_ne_masks': neg_ne_masks,
            # Add the raw texts
            'src_text': src_text,
            'pos_texts': pos_texts,
            'neg_texts': neg_texts,
        }

        return sample


def contrastive_collate_fn(samples):
    if len(samples) == 0:
        return {}

    # Batch source sentences
    src_input_ids = torch.stack([s['src_input_ids'] for s in samples])
    src_attention_mask = torch.stack([s['src_attention_mask'] for s in samples])

    batch_size = src_input_ids.size(0)

    # Concatenate positive and negative targets
    contrast_input_ids_list = []
    contrast_ne_masks_list = []
    src_select_index = []
    cross_entropy_pos = []
    accumulate_cnt = 0

    for i, s in enumerate(samples):
        # Positive targets
        pos_input_ids_list = s['pos_input_ids_list']
        pos_ne_masks = s['pos_ne_masks']
        num_pos = len(pos_input_ids_list)
        contrast_input_ids_list.extend(pos_input_ids_list)
        contrast_ne_masks_list.extend(pos_ne_masks)
        src_select_index.extend([i] * num_pos)
        accumulate_cnt += num_pos
        cross_entropy_pos.append(accumulate_cnt - 1)  # Last positive target

        # Negative targets
        neg_input_ids_list = s['neg_input_ids_list']
        neg_ne_masks = s['neg_ne_masks']
        num_neg = len(neg_input_ids_list)
        contrast_input_ids_list.extend(neg_input_ids_list)
        contrast_ne_masks_list.extend(neg_ne_masks)
        src_select_index.extend([i] * num_neg)
        accumulate_cnt += num_neg

    # Stack contrast targets
    contrast_input_ids = torch.stack(contrast_input_ids_list)
    contrast_ne_masks = torch.stack(contrast_ne_masks_list)
    src_select_index = torch.tensor(src_select_index, dtype=torch.long)

    # Prepare contrast_target_input_ids (the last positive target from each sample)
    contrast_target_input_ids = torch.stack([
        s['pos_input_ids_list'][-1] for s in samples
    ])

    # Prepare valid_contrast and positive_contrast matrices
    total_contrast_samples = contrast_input_ids.size(0)
    valid_contrast = torch.zeros((total_contrast_samples, total_contrast_samples), dtype=torch.bool)
    positive_contrast = torch.zeros((total_contrast_samples, total_contrast_samples), dtype=torch.bool)

    accumulate_cnt = 0
    for s in samples:
        num_pos = len(s['pos_input_ids_list'])
        num_neg = len(s['neg_input_ids_list'])

        # Indices for this sample
        indices = torch.arange(accumulate_cnt, accumulate_cnt + num_pos + num_neg)
        pos_indices = torch.arange(accumulate_cnt, accumulate_cnt + num_pos)
        neg_indices = torch.arange(accumulate_cnt + num_pos, accumulate_cnt + num_pos + num_neg)

        # Valid contrasts include all combinations within this sample
        valid_contrast[indices.unsqueeze(0), indices.unsqueeze(1)] = True
        # Positive contrasts are between positive targets
        positive_contrast[pos_indices.unsqueeze(0), pos_indices.unsqueeze(1)] = True

        accumulate_cnt += num_pos + num_neg

    # Remove self-contrast
    valid_contrast.fill_diagonal_(False)
    positive_contrast.fill_diagonal_(False)

    batch = {
        'src_input_ids': src_input_ids,
        'src_attention_mask': src_attention_mask,
        'contrast_input_ids': contrast_input_ids,
        'contrast_ne_masks': contrast_ne_masks,
        'contrast_target_input_ids': contrast_target_input_ids,
        'src_select_index': src_select_index,
        'valid_contrast': valid_contrast,
        'positive_contrast': positive_contrast,
        'cross_entropy_pos': cross_entropy_pos,
    }

    return batch
