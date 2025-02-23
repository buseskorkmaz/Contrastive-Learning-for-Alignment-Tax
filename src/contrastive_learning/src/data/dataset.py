# src/data/dataset.py - Dataset and collator for contrastive learning
from typing import Dict, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np
from collections import defaultdict
import logging
import os

class ContrastiveDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 256,
        pos_data: str = None,
        neg_data: str = None,
        neg_types: List[str] = None,
        max_neg_samples: int = 5,
        seed: int = 42,
        epoch: int = 1,
        logger = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neg_samples = max_neg_samples
        self.epoch = epoch - 1
        self.logger = logger or logging.getLogger(__name__)
        self.neg_types = neg_types or []
        
        try:
            model_name = model_name_or_path.split("/")[1]
        except:
            model_name = "gpt2"

        # First collect all possible indices with positive samples
        self.pos_samples = defaultdict(list)
        self.pos_ne = defaultdict(list)
        indices_with_pos = set()
        
        # Load original positives
        with open(os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.pos_target")) as f_tgt, \
            open(os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.pos_ne")) as f_ne, \
            open(os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.other")) as f_other:
            for i, (tgt_line, ne_line, other_line) in enumerate(zip(f_tgt, f_ne, f_other)):
                idx = int(other_line.split('\t')[0])
                self.pos_samples[idx].append(tgt_line.strip())
                self.pos_ne[idx].append([int(x) for x in ne_line.strip().split()])
                indices_with_pos.add(idx)
        
        # Load back-translated positives if available
        if os.path.exists(os.path.join(pos_data, "back_translation")):
            with open(os.path.join(pos_data, "back_translation", f"{model_name}/{split}/{split}.pos_target")) as f_tgt, \
                open(os.path.join(pos_data, "back_translation", f"{model_name}/{split}/{split}.pos_ne")) as f_ne, \
                open(os.path.join(pos_data, "back_translation", f"{model_name}/{split}/{split}.other")) as f_other:
                for i, (tgt_line, ne_line, other_line) in enumerate(zip(f_tgt, f_ne, f_other)):
                    idx = int(other_line.split('\t')[0])
                    self.pos_samples[idx].append(tgt_line.strip())
                    self.pos_ne[idx].append([int(x) for x in ne_line.strip().split()])
                    indices_with_pos.add(idx)

        # Then collect all indices with negative samples
        self.neg_samples = defaultdict(list)
        self.neg_ne = defaultdict(list)
        self.loaded_neg_types = set()
        indices_with_neg = set()
        
        if neg_types and neg_data:
            for neg_type in neg_types:
                neg_type_dir = os.path.join(neg_data, neg_type)
                if not os.path.exists(neg_type_dir):
                    self.logger.warning(f"Negative type directory not found: {neg_type_dir}")
                    continue
                    
                try:
                    with open(os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.neg_target")) as f_tgt, \
                         open(os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.neg_ne")) as f_ne, \
                         open(os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.other")) as f_other:
                        for tgt_line, ne_line, other_line in zip(f_tgt, f_ne, f_other):
                            idx, bpe_start = map(int, other_line.strip().split('\t'))
                            self.neg_samples[idx].append(tgt_line.strip())
                            self.neg_ne[idx].append([int(x) for x in ne_line.strip().split()])
                            indices_with_neg.add(idx)
                    self.loaded_neg_types.add(neg_type)
                except Exception as e:
                    self.logger.error(f"Error loading negative type {neg_type}: {str(e)}")

        # Find indices that have both positives and negatives
        valid_indices = indices_with_pos & indices_with_neg
        
        if not valid_indices:
            raise ValueError("No samples found with both positive and negative examples!")
        
        self.logger.info(f"Found {len(valid_indices)} samples with both positive and negative examples")
        self.logger.info(f"Original samples with positives: {len(indices_with_pos)}")
        self.logger.info(f"Original samples with negatives: {len(indices_with_neg)}")
        
        # Load sources only for valid indices
        with open(os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.source")) as f:
            all_sources = [line.strip() for line in f]
            self.sources = [all_sources[idx] for idx in valid_indices]

        # Filter pos_samples and neg_samples to only include valid indices
        self.pos_samples = {idx: self.pos_samples[idx] for idx in valid_indices}
        self.pos_ne = {idx: self.pos_ne[idx] for idx in valid_indices}
        self.neg_samples = {idx: self.neg_samples[idx] for idx in valid_indices}
        self.neg_ne = {idx: self.neg_ne[idx] for idx in valid_indices}
        
        # Create shuffle for negatives
        self.neg_shuffle = {
            idx: np.random.RandomState(seed).permutation(len(samples))
            for idx, samples in self.neg_samples.items()
        }
        
        # Store valid indices as a sorted list for deterministic access
        self.indices = sorted(valid_indices)
        
        # Create index mapping for efficient access
        self.idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.indices)}
        
        # Validate final dataset
        self.validate_dataset()
        
        # Log final dataset statistics
        self.logger.info(f"Final dataset size: {len(self.indices)}")
        avg_pos = sum(len(pos) for pos in self.pos_samples.values()) / len(self.indices)
        avg_neg = sum(len(neg) for neg in self.neg_samples.values()) / len(self.indices)
        self.logger.info(f"Average positives per sample: {avg_pos:.2f}")
        self.logger.info(f"Average negatives per sample: {avg_neg:.2f}")

    def __getitem__(self, idx):
        """Get item using new index mapping"""
        true_idx = self.indices[idx]
        source = self.sources[idx]  # sources already filtered
        
        # Get positive samples
        pos_targets = self.pos_samples[true_idx]
        pos_ne = self.pos_ne[true_idx]
        
        # Get negative samples with epoch-aware sampling
        neg_targets = []
        neg_ne = []
        if len(self.neg_samples[true_idx]) <= self.max_neg_samples:
            neg_targets = self.neg_samples[true_idx]
            neg_ne = self.neg_ne[true_idx]
        else:
            start_idx = self.epoch * self.max_neg_samples
            indices = [
                self.neg_shuffle[true_idx][i % len(self.neg_samples[true_idx])]
                for i in range(start_idx, start_idx + self.max_neg_samples)
            ]
            neg_targets = [self.neg_samples[true_idx][i] for i in indices]
            neg_ne = [self.neg_ne[true_idx][i] for i in indices]
        
        return {
            "id": true_idx,
            "source": source,
            "pos_target": pos_targets,
            "pos_ne": pos_ne,
            "neg_target": neg_targets,
            "neg_ne": neg_ne
        }

    def __len__(self):
        return len(self.indices)

    def validate_dataset(self):
        """Validate final dataset"""
        total_samples = len(self.indices)
        if total_samples == 0:
            raise ValueError("No valid samples in dataset!")
            
        # Verify all samples have both positives and negatives
        for idx in self.indices:
            if not self.pos_samples[idx]:
                raise ValueError(f"Sample {idx} missing positive examples!")
            if not self.neg_samples[idx]:
                raise ValueError(f"Sample {idx} missing negative examples!")
        
        self.logger.info(f"Dataset Statistics for {self.__class__.__name__}:")
        self.logger.info(f"Total valid samples: {total_samples}")
        self.logger.info(f"Loaded negative types: {list(self.loaded_neg_types)}")
        

from collections import defaultdict
import torch
from typing import List, Dict
from transformers import PreTrainedTokenizer

class ContrastiveCollator:
    def __init__(
        self,
        logger, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        pad_to_multiple_of: int = 8
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.logger = logger
        self.max_positives_per_feature=2

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Add max positives parameter
        max_positives_per_feature = 2  # You can make this a parameter in __init__
        
        # Initial logging
        self.logger.info(f"Initial number of features: {len(features)}")
        self.logger.info(f"Number of positive samples per feature: {[len(f['pos_target']) for f in features]}")
        self.logger.info(f"Number of negative samples per feature: {[len(f.get('neg_target', [])) for f in features]}")

        # Track sample positions and type
        sample_positions = []
        accumulate_cnt = 0
        
        self.logger.info("Creating initial index mapping...")
        for feature in features:
            # Limit number of positive samples
            pos_targets = feature["pos_target"][:max_positives_per_feature]
            pos_count = len(pos_targets)
            neg_count = len(feature.get("neg_target", []))
            
            if pos_count > 0:
                sample_positions.append((accumulate_cnt, accumulate_cnt + pos_count, True))
                accumulate_cnt += pos_count
                
            if neg_count > 0:
                sample_positions.append((accumulate_cnt, accumulate_cnt + neg_count, False))
                accumulate_cnt += neg_count

        self.logger.info(f"Sample positions created: {sample_positions}")

        # Prepare sources and collect all targets
        sources = [f["source"] for f in features]
        all_targets = []
        all_ne = []
        src_indices = []
        all_aug_types = []  
        total_positives = 0
        total_negatives = 0

        # Process each feature's targets with limited positives
        for i, feature in enumerate(features):
            # Add limited positive samples
            pos_targets = feature["pos_target"][:max_positives_per_feature]
            pos_ne = feature["pos_ne"][:max_positives_per_feature]
            for _ in pos_targets:
                all_aug_types.append("pos/original")
            for pos_tgt, p_ne in zip(pos_targets, pos_ne):
                all_targets.append(pos_tgt)
                all_ne.append(p_ne)
                src_indices.append(i)
                total_positives += 1

            # Add negative samples if they exist
            if "neg_target" in feature and feature["neg_target"]:
                for neg_tgt, neg_ne, aug in zip(feature["neg_target"], feature["neg_ne"], feature.get("augmentation_types", [])):
                    all_targets.append(neg_tgt)
                    all_ne.append(neg_ne)
                    src_indices.append(i)
                    all_aug_types.append(aug)
                    total_negatives += 1

        self.logger.info(f"Total positive samples (limited): {total_positives}")
        self.logger.info(f"Total negative samples: {total_negatives}")
        self.logger.info(f"Source indices: {src_indices[:20]}")  # Debug first 20 source indices

        # Create input texts and process them
        input_texts = []
        labels_list = []
        adjusted_ne = []
        valid_indices = []

        for idx, (target_text, ne, src_idx) in enumerate(zip(all_targets, all_ne, src_indices)):
            source_text = sources[src_idx]

            # Tokenize with length constraints
            max_source_length = int(self.max_length * 0.6)
            max_target_length = self.max_length - max_source_length

            source_encoding = self.tokenizer(
                source_text,
                truncation=True,
                max_length=max_source_length,
                add_special_tokens=False,
            )
            target_encoding = self.tokenizer(
                target_text,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )

            source_len = len(source_encoding['input_ids'])
            target_len = len(target_encoding['input_ids'])

            # Skip if target is empty after truncation
            if target_len == 0:
                self.logger.warning(f"Sample {idx} skipped: empty target after truncation")
                continue

            # Combine source and target
            input_ids = source_encoding['input_ids'] + target_encoding['input_ids']
            input_ids = input_ids[:self.max_length]

            # Create labels with -100 for source tokens
            labels = [-100] * source_len + target_encoding['input_ids']
            labels = labels[:len(input_ids)]

            # Adjust NE sequence
            ne = ne[:target_len]
            ne_padded = ne + [0] * (target_len - len(ne))
            adjusted_ne_seq = [0] * source_len + ne_padded
            adjusted_ne_seq = adjusted_ne_seq[:len(input_ids)]

            # Add to lists
            input_texts.append(self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))
            labels_list.append(labels)
            adjusted_ne.append(adjusted_ne_seq)
            valid_indices.append(idx)

        # Check if we have any valid samples
        if not valid_indices:
            self.logger.error("No valid samples after processing")
            return {}

        # Create batch encoding
        batch_encoding = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=False,
        )

        # Create mapping from original to valid indices
        valid_pos_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
        self.logger.info(f"Valid position mapping: {valid_pos_map}")

        # Create source to positives mapping
        source_to_positives = defaultdict(list)
        
        self.logger.info("Creating source to positives mapping...")
        for start_idx, end_idx, is_positive in sample_positions:
            if is_positive:
                source_idx = src_indices[start_idx]
                valid_idxs = [i for i in range(start_idx, end_idx) if i in valid_pos_map]
                if valid_idxs:
                    source_to_positives[source_idx].extend(valid_pos_map[i] for i in valid_idxs)

        # Create contrast matrices
        total_samples = len(valid_indices)
        valid_contrast = torch.ones((total_samples, total_samples), dtype=torch.bool)
        positive_contrast = torch.zeros((total_samples, total_samples), dtype=torch.bool)

        # Fill positive contrasts by source
        self.logger.info("Creating contrast matrices...")
        self.logger.info(f"Source to positives mapping: {dict(source_to_positives)}")
        
        for source_idx, positive_indices in source_to_positives.items():
            for i in positive_indices:
                for j in positive_indices:
                    if i != j:
                        positive_contrast[i, j] = True

        # Remove self-contrasts
        for i in range(total_samples):
            valid_contrast[i, i] = False
            positive_contrast[i, i] = False

        pos_sum = positive_contrast.sum().item()
        self.logger.info(f"Positive contrast matrix sum: {pos_sum}")
        self.logger.info(f"Valid contrast matrix sum: {valid_contrast.sum().item()}")

        if pos_sum == 0:
            self.logger.error("No positive contrasts found!")
            self.logger.error(f"Source to positives mapping: {dict(source_to_positives)}")
            raise ValueError("No positive contrasts could be created from the batch")

        # Create final tensors
        max_seq_len = batch_encoding['input_ids'].size(1)
        padded_labels = self._pad_sequences(labels_list, max_seq_len, -100)
        padded_ne = self._pad_sequences(adjusted_ne, max_seq_len, 0)

        # Final assertions and checks
        assert batch_encoding['input_ids'].size(0) == len(valid_indices), \
            f"Batch size mismatch: {batch_encoding['input_ids'].size(0)} vs {len(valid_indices)}"
        assert positive_contrast.sum() > 0, "No positive contrasts in batch"
        assert valid_contrast.sum() > 0, "No valid contrasts in batch"

        # Log final statistics
        self.logger.info(f"Final batch size: {len(valid_indices)}")
        self.logger.info(f"Number of positive contrasts: {positive_contrast.sum().item() // 2}")
        self.logger.info(f"Number of valid contrasts: {valid_contrast.sum().item() // 2}")

        return {
            "net_input": {
                "input_ids": batch_encoding['input_ids'],
                "attention_mask": batch_encoding['attention_mask'],
            },
            "target": torch.tensor(padded_labels),
            "contrast_src_select_index": torch.tensor(src_indices, dtype=torch.long),
            "valid_contrast": valid_contrast,
            "positive_contrast": positive_contrast,
            "contrast_ne": torch.tensor(padded_ne),
            "ntokens": (torch.tensor(padded_labels) != -100).sum().item(),
            "augmentation_types": all_aug_types
        }

    def _pad_sequences(self, sequences, max_len, pad_value):
        """Helper function to pad sequences to the same length"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padded.append(seq + [pad_value] * (max_len - len(seq)))
            else:
                padded.append(seq[:max_len])
        return padded

# def prepare_data_for_training(args, model_name_or_path, tokenizer, accelerator, logger, max_eval_samples: Optional[int] = 100):
#     """
#     Prepare datasets and collator with an option to limit evaluation dataset size.
#     """
#     logger.info("Loading datasets...")
    
#     # Setup paths
#     pos_data = os.path.join(args.data_path, args.pos_data_dir)
#     neg_data = os.path.join(args.data_path, args.neg_data_dir)
    
#     # Create training dataset
#     train_dataset = ContrastiveDataset(
#         model_name_or_path=model_name_or_path,
#         data_path=args.data_path,
#         tokenizer=tokenizer,
#         split="train",
#         max_length=args.max_length,
#         pos_data=pos_data,
#         neg_data=neg_data,
#         neg_types=args.neg_types,
#         max_neg_samples=args.max_neg_samples,
#         epoch=1  # Will be updated during training
#     )
    
#     # Create validation dataset if needed
#     eval_dataset = None
#     if args.validation_split:
#         eval_dataset = ContrastiveDataset(
#             model_name_or_path=model_name_or_path,
#             data_path=args.data_path,
#             tokenizer=tokenizer,
#             split=args.validation_split,
#             max_length=args.max_length,
#             pos_data=pos_data,
#             neg_data=neg_data,
#             neg_types=args.neg_types,
#             max_neg_samples=args.max_neg_samples,
#             epoch=1
#         )
        
#         # Limit the number of evaluation samples if max_eval_samples is specified
#         if max_eval_samples is not None:
#             eval_dataset = torch.utils.data.Subset(eval_dataset, range(min(len(eval_dataset), max_eval_samples)))
#             logger.info(f"Using only {len(eval_dataset)} samples for evaluation.")

#     # Create collator
#     collator = SmartContrastiveCollator(
#         logger=logger,
#         tokenizer=tokenizer,
#         max_length=args.max_length,
#         pad_to_multiple_of=8  # For efficient GPU usage
#     )
    
#     logger.info(f"Loaded {len(train_dataset)} training examples")
#     if eval_dataset:
#         logger.info(f"Loaded {len(eval_dataset)} validation examples")
    
#     return train_dataset, eval_dataset, collator
import torch
from torch.utils.data import random_split, Subset

def prepare_data_for_training(
    args, model_name_or_path, tokenizer, accelerator, logger, max_eval_samples: Optional[int] = 32
):
    logger.info("Loading full dataset from 'train' split...")

    # Load the entire train dataset
    full_dataset = ContrastiveDataset(
        model_name_or_path=model_name_or_path,
        data_path=args.data_path,
        tokenizer=tokenizer,
        split="train",  # Load the 'train' portion
        max_length=args.max_length,
        pos_data=os.path.join(args.data_path, args.pos_data_dir),
        neg_data=os.path.join(args.data_path, args.neg_data_dir),
        neg_types=args.neg_types,
        max_neg_samples=args.max_neg_samples,
        epoch=1
    )

    logger.info(f"Loaded {len(full_dataset)} total training samples.")
    
    # Decide how many samples go to training vs. validation
    train_size = len(full_dataset) - max_eval_samples
    val_size = max_eval_samples
    
    # Randomly split into train_dataset and eval_dataset
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])
    
    logger.info(f"Using {train_size} samples for training, {val_size} samples for validation.")
    
    # Optionally limit the number of validation samples
    if max_eval_samples is not None:
        eval_dataset = Subset(eval_dataset, range(min(len(eval_dataset), max_eval_samples)))
        logger.info(f"Using only {len(eval_dataset)} samples for validation (limited).")
    
    # Create collator (use whichever collator class you prefer)
    collator = SmartContrastiveCollator(
        logger=logger,
        tokenizer=tokenizer,
        max_length=args.max_length,
        pad_to_multiple_of=8
    )
    
    logger.info("Inspecting dataset samples...")
    inspect_data_sample(train_dataset, collator, tokenizer, num_samples=2)
    
    return train_dataset, eval_dataset, collator

def inspect_data_sample(dataset, collator, tokenizer, num_samples=2):
    """Detailed inspection of dataset samples and batches"""
    print("\n=== Dataset Inspection ===")
    
    # 1. Look at raw samples
    print("\n1. Raw Sample Inspection:")
    for i in range(num_samples):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"Source text: {sample['source'][:300]}...")
        
        print("\nPositive samples:")
        for j, (pos_tgt, pos_ne) in enumerate(zip(sample['pos_target'], sample['pos_ne'])):
            ne_tokens = sum(1 for x in pos_ne if x > 0)
            print(f"  Pos {j}: {pos_tgt[:300]}...")
            print(f"  NE tokens: {ne_tokens}")
            if ne_tokens > 0:
                # Show the actual NE tokens
                tokenized = tokenizer(pos_tgt, add_special_tokens=False)
                ne_text = [tokenizer.decode([tokenized['input_ids'][idx]]) 
                          for idx, val in enumerate(pos_ne) if val > 0]
                print(f"  NE text: {ne_text}")
        
        print("\nNegative samples:")
        for j, (neg_tgt, neg_ne) in enumerate(zip(sample['neg_target'], sample['neg_ne'])):
            ne_tokens = sum(1 for x in neg_ne if x > 0)
            print(f"  Neg {j}: {neg_tgt[:300]}...")
            print(f"  NE tokens: {ne_tokens}")
            if ne_tokens > 0:
                tokenized = tokenizer(neg_tgt, add_special_tokens=False)
                ne_text = [tokenizer.decode([tokenized['input_ids'][idx]]) 
                          for idx, val in enumerate(neg_ne) if val > 0]
                print(f"  NE text: {ne_text}")
    
    # 2. Look at collated batches
    print("\n2. Batch Statistics:")
    loader = DataLoader(dataset, batch_size=4, collate_fn=collator, shuffle=True)
    batch = next(iter(loader))
    
    print("\nBatch shapes:")
    print(f"Input IDs: {batch['net_input']['input_ids'].shape}")
    print(f"Attention Mask: {batch['net_input']['attention_mask'].shape}")
    print(f"Labels: {batch['target'].shape}")
    print(f"NE Mask: {batch['contrast_ne'].shape}")
    
    print("\nNE statistics:")
    ne_mask = batch['contrast_ne']
    print(f"Total NE tokens: {ne_mask.sum().item()}")
    print(f"NE tokens per sequence: {ne_mask.sum(dim=1).tolist()}")
    print(f"Average NE tokens per sequence: {ne_mask.sum(dim=1).float().mean().item():.2f}")
    
    print("\nContrast pair statistics:")
    print(f"Valid contrasts: {batch['valid_contrast'].sum().item()}")
    print(f"Positive contrasts: {batch['positive_contrast'].sum().item()}")
    print(f"Contrast matrix:")
    print(batch['positive_contrast'].int())
    
    # 3. Look at actual token content
    print("\n3. Token Content Analysis:")
    b_idx = 0  # Look at first sequence in batch
    input_ids = batch['net_input']['input_ids'][b_idx]
    ne_mask = batch['contrast_ne'][b_idx]
    
    # Decode and show NE tokens
    ne_positions = ne_mask.nonzero().squeeze(-1)
    if len(ne_positions) > 0:
        print("\nNamed Entity tokens:")
        for pos in ne_positions:
            token = input_ids[pos]
            token_text = tokenizer.decode([token])
            print(f"Position {pos}: {token_text}")
    
    return batch  # Return a batch for further inspection if needed


from collections import defaultdict
import torch
import random
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer

class SmartContrastiveCollator:
    def __init__(
        self,
        logger, 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,  # Make sure this matches your model's max sequence length
        pad_to_multiple_of: int = 8,
        max_positives_per_feature: int = 2,
        max_resampling_attempts: int = 10,
        min_negatives_per_batch: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.logger = logger
        self.max_positives_per_feature = max_positives_per_feature
        self.max_resampling_attempts = max_resampling_attempts
        self.min_negatives_per_batch = min_negatives_per_batch

    def process_batch(self, features: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        """Process a single batch of features, return None if invalid"""
        self.logger.info(f"Processing batch with {len(features)} features")
        self.logger.info(f"Positive samples per feature: {[len(f['pos_target']) for f in features]}")
        self.logger.info(f"Negative samples per feature: {[len(f.get('neg_target', [])) for f in features]}")

        # Early check for minimum negative samples across all features
        total_neg_samples = sum(len(f.get('neg_target', [])) for f in features)
        if total_neg_samples < self.min_negatives_per_batch:
            self.logger.warning(f"Insufficient negative samples in batch: {total_neg_samples} < {self.min_negatives_per_batch}")
            return None

        # Track sample positions and type
        sample_positions = []
        accumulate_cnt = 0
        total_positives = 0
        total_negatives = 0

        # Map features to positions
        for feature in features:
            pos_targets = feature["pos_target"][:self.max_positives_per_feature]
            pos_count = len(pos_targets)
            neg_count = len(feature.get("neg_target", []))
            
            if pos_count > 0:
                sample_positions.append((accumulate_cnt, accumulate_cnt + pos_count, True))
                accumulate_cnt += pos_count
                total_positives += pos_count
                
            if neg_count > 0:
                sample_positions.append((accumulate_cnt, accumulate_cnt + neg_count, False))
                accumulate_cnt += neg_count
                total_negatives += neg_count

        # Verify minimum requirements
        if total_positives < 2 or total_negatives < self.min_negatives_per_batch:
            self.logger.warning(f"Batch requirements not met: positives={total_positives}, negatives={total_negatives}")
            return None

        # Collect all targets
        sources = [f["source"] for f in features]
        all_targets = []
        all_ne = []
        src_indices = []
        
        for i, feature in enumerate(features):
            # Add limited positive samples
            pos_targets = feature["pos_target"][:self.max_positives_per_feature]
            pos_ne = feature["pos_ne"][:self.max_positives_per_feature]
            
            for pos_tgt, p_ne in zip(pos_targets, pos_ne):
                all_targets.append(pos_tgt)
                all_ne.append(p_ne)
                src_indices.append(i)

            # Add negative samples
            if "neg_target" in feature and feature["neg_target"]:
                for neg_tgt, neg_ne in zip(feature["neg_target"], feature["neg_ne"]):
                    all_targets.append(neg_tgt)
                    all_ne.append(neg_ne)
                    src_indices.append(i)

        # Process texts
        input_texts = []
        labels_list = []
        adjusted_ne = []
        valid_indices = []

        for idx, (target_text, ne, src_idx) in enumerate(zip(all_targets, all_ne, src_indices)):
            source_text = sources[src_idx]
            
            # Tokenize with balanced length constraints
            max_source_length = int(self.max_length * 0.6)
            max_target_length = self.max_length - max_source_length

            source_encoding = self.tokenizer(
                source_text,
                truncation=True,
                max_length=max_source_length,
                add_special_tokens=False,
            )
            target_encoding = self.tokenizer(
                target_text,
                truncation=True,
                max_length=max_target_length,
                add_special_tokens=False,
            )

            source_len = len(source_encoding['input_ids'])
            target_len = len(target_encoding['input_ids'])

            # Skip invalid samples
            if target_len == 0:
                continue

            # Create combined input and labels
            input_ids = source_encoding['input_ids'] + target_encoding['input_ids']
            input_ids = input_ids[:self.max_length]
            labels = [-100] * source_len + target_encoding['input_ids']
            labels = labels[:len(input_ids)]

            # Process NE sequence
            ne = ne[:target_len]
            ne_padded = ne + [0] * (target_len - len(ne))
            adjusted_ne_seq = [0] * source_len + ne_padded
            adjusted_ne_seq = adjusted_ne_seq[:len(input_ids)]

            input_texts.append(self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))
            labels_list.append(labels)
            adjusted_ne.append(adjusted_ne_seq)
            valid_indices.append(idx)

        if not valid_indices:
            return None

        # Create tensors
        # When creating batch_encoding, explicitly use max_length
        batch_encoding = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,  # Make sure this is enforced
            return_tensors='pt',
            add_special_tokens=False,
        )

        # Add debug logging for shapes
        self.logger.info(f"Batch encoding input_ids shape: {batch_encoding['input_ids'].shape}")

        # Create contrast matrices
        valid_pos_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(valid_indices)}
        source_to_positives = defaultdict(list)
        
        for start_idx, end_idx, is_positive in sample_positions:
            if is_positive:
                source_idx = src_indices[start_idx]
                valid_idxs = [i for i in range(start_idx, end_idx) if i in valid_pos_map]
                if valid_idxs:
                    source_to_positives[source_idx].extend(valid_pos_map[i] for i in valid_idxs)

        total_samples = len(valid_indices)
        valid_contrast = torch.ones((total_samples, total_samples), dtype=torch.bool)
        positive_contrast = torch.zeros((total_samples, total_samples), dtype=torch.bool)

        for source_idx, positive_indices in source_to_positives.items():
            for i in positive_indices:
                for j in positive_indices:
                    if i != j:
                        positive_contrast[i, j] = True

        # Remove self-contrasts
        for i in range(total_samples):
            valid_contrast[i, i] = False
            positive_contrast[i, i] = False

        if positive_contrast.sum().item() == 0:
            return None

        # Create final tensors
        max_seq_len = min(self.max_length, batch_encoding['input_ids'].size(1))
        
        # Ensure all sequences are properly truncated/padded to max_seq_len
        padded_labels = self._pad_sequences(labels_list, max_seq_len, -100)
        padded_ne = []
        for ne_seq in adjusted_ne:
            if len(ne_seq) < max_seq_len:
                padded_ne.append(ne_seq + [0] * (max_seq_len - len(ne_seq)))
            else:
                padded_ne.append(ne_seq[:max_seq_len])

        # Add shape verification logging
        padded_ne_tensor = torch.tensor(padded_ne)
        self.logger.info(f"Padded NE tensor shape: {padded_ne_tensor.shape}")
        self.logger.info(f"Input IDs shape: {batch_encoding['input_ids'].shape}")

        return {
            "net_input": {
                "input_ids": batch_encoding['input_ids'][:, :max_seq_len],
                "attention_mask": batch_encoding['attention_mask'][:, :max_seq_len],
            },
            "target": torch.tensor(padded_labels),
            "contrast_src_select_index": torch.tensor([src_indices[i] for i in valid_indices], dtype=torch.long),
            "valid_contrast": valid_contrast,
            "positive_contrast": positive_contrast,
            "contrast_ne": torch.tensor(padded_ne),
            "ntokens": (torch.tensor(padded_labels) != -100).sum().item()
        }

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Main collation function that strictly requires negative samples"""
        base_features = features.copy()
        attempted_combinations = set()
        
        # Early check for features with negatives
        features_with_neg = [f for f in base_features if len(f.get('neg_target', [])) > 0]
        
        if not features_with_neg:
            raise RuntimeError(
                "No features with negative samples in batch. "
                "Contrastive learning requires negative samples. "
                "Consider modifying dataset to ensure negative samples in each batch."
            )
        
        for attempt in range(self.max_resampling_attempts):
            self.logger.info(f"Processing attempt {attempt + 1}/{self.max_resampling_attempts}")
            
            if attempt > 0:
                # Create new combination ensuring at least one negative
                neg_feature = random.choice(features_with_neg)
                features = [neg_feature]
                
                # Add other features if available
                remaining_features = [f for f in base_features if f != neg_feature]
                if remaining_features:
                    num_additional = min(len(remaining_features), 
                                    random.randint(1, min(3, len(base_features)-1)))
                    features.extend(random.sample(remaining_features, num_additional))
                
                random.shuffle(features)
                
                # Verify we still have required negatives
                if sum(len(f.get('neg_target', [])) for f in features) < self.min_negatives_per_batch:
                    continue
            
            # Try processing current combination
            result = self.process_batch(features)
            if result is not None:
                self.logger.info(f"Successfully created batch with negatives")
                return result

        # If we get here, all attempts failed
        raise RuntimeError(
            f"Failed to create valid batch after {self.max_resampling_attempts} attempts. "
            f"Total features: {len(base_features)}, "
            f"Features with negatives: {len(features_with_neg)}. "
            "Increase batch size or modify dataset to ensure sufficient negative samples."
        )

    def _pad_sequences(self, sequences, max_len, pad_value):
        """Helper function to pad sequences to the same length"""
        padded = []
        for seq in sequences:
            if len(seq) < max_len:
                padded.append(seq + [pad_value] * (max_len - len(seq)))
            else:
                padded.append(seq[:max_len])
        return padded


# class ContrastiveCollator:
#     def __init__(
#         self,
#         tokenizer: PreTrainedTokenizer,
#         max_length: int = 1024,
#         pad_to_multiple_of: int = 8
#     ):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.pad_token_id = tokenizer.pad_token_id
#         self.eos_token_id = tokenizer.eos_token_id
        
#     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
#         # Prepare inputs exactly like fairseq
#         sources = [f["source"] for f in features]
        
#         # Collect all targets maintaining order
#         all_targets = []
#         all_ne = []
#         src_indices = []
#         pos_positions = []  # For CE loss
        
#         valid_contrasts = []
#         positive_contrasts = []
        
#         accumulate = 0
#         for i, feature in enumerate(features):
#             # Add positive samples first
#             valid_i = []
#             for pos_tgt, pos_ne in zip(feature["pos_target"], feature["pos_ne"]):
#                 all_targets.append(pos_tgt)
#                 all_ne.append(pos_ne)
#                 src_indices.append(i)
                
#                 if not all(x == 0 for x in pos_ne):
#                     valid_i.append(accumulate)
#                 accumulate += 1
            
#             # Track cross entropy position
#             if valid_i:
#                 pos_positions.append(valid_i[-1])
            
#             # Add valid contrasts for positives
#             for idx1, idx2 in itertools.combinations(valid_i, 2):
#                 positive_contrasts.append((idx1, idx2))
#                 valid_contrasts.append((idx1, idx2))
            
#             # Then add negative samples
#             valid_j = []
#             for neg_tgt, neg_ne in zip(feature["neg_target"], feature["neg_ne"]):
#                 all_targets.append(neg_tgt)
#                 all_ne.append(neg_ne)
#                 src_indices.append(i)
#                 valid_j.append(accumulate)
#                 accumulate += 1
                
#             # Add valid contrasts for negatives
#             for idx1, idx2 in itertools.combinations(valid_j, 2):
#                 valid_contrasts.append((idx1, idx2))
        
#         # Tokenize all inputs
#         source_tokens = self.tokenizer(
#             sources,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         target_tokens = self.tokenizer(
#             all_targets,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt"
#         )
        
#         # Properly pad and convert NE sequences
#         max_ne_len = max(len(ne) for ne in all_ne)
#         padded_ne = [
#             ne + [0] * (max_ne_len - len(ne))  # Pad with zeros
#             for ne in all_ne
#         ]
#         ne_tensor = torch.tensor(padded_ne, dtype=torch.long)
        
#         # Match NE tensor length with target sequence length
#         if ne_tensor.size(1) < target_tokens.input_ids.size(1):
#             # If NE sequence is shorter, pad with zeros
#             padding = torch.zeros(
#                 (ne_tensor.size(0), target_tokens.input_ids.size(1) - ne_tensor.size(1)),
#                 dtype=ne_tensor.dtype
#             )
#             ne_tensor = torch.cat([ne_tensor, padding], dim=1)
#         elif ne_tensor.size(1) > target_tokens.input_ids.size(1):
#             # If NE sequence is longer, truncate
#             ne_tensor = ne_tensor[:, :target_tokens.input_ids.size(1)]
        
#         # Create contrast matrices
#         total_targets = len(all_targets)
#         valid_contrast_matrix = torch.zeros((total_targets, total_targets), dtype=torch.bool)
#         positive_contrast_matrix = torch.zeros((total_targets, total_targets), dtype=torch.bool)
        
#         for i, j in valid_contrasts:
#             valid_contrast_matrix[i, j] = True
#             valid_contrast_matrix[j, i] = True
            
#         for i, j in positive_contrasts:
#             positive_contrast_matrix[i, j] = True
#             positive_contrast_matrix[j, i] = True
            
#         # Remove self-contrasts
#         for i in range(total_targets):
#             valid_contrast_matrix[i, i] = False
#             positive_contrast_matrix[i, i] = False
            
#         return {
#             "net_input": {
#                 "input_ids": source_tokens.input_ids,
#                 "attention_mask": source_tokens.attention_mask,
#                 "labels": None  # We'll handle loss calculation ourselves, not through LLaMA's loss
#             },
#             "target": target_tokens.input_ids,
#             "ce_pos": torch.tensor(pos_positions, dtype=torch.long),
#             "contrast_src_select_index": torch.tensor(src_indices, dtype=torch.long),
#             "valid_contrast": valid_contrast_matrix,
#             "positive_contrast": positive_contrast_matrix,
#             "contrast_ne": ne_tensor,
#             "ntokens": (target_tokens.input_ids != self.pad_token_id).sum().item()
#         }

# class ContrastiveCollator:
#     def __init__(
#         self,
#         logger, 
#         tokenizer: PreTrainedTokenizer,
#         max_length: int = 256,
#         pad_to_multiple_of: int = 8
#     ):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.pad_token_id = tokenizer.pad_token_id
#         self.eos_token_id = tokenizer.eos_token_id
#         self.logger = logger

#     def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
#         # Prepare the source texts
#         sources = [f["source"] for f in features]
#         # Log initial features
#         self.logger.info(f"Initial number of features: {len(features)}")
#         self.logger.info(f"Number of positive samples per feature: {[len(f['pos_target']) for f in features]}")
#         self.logger.info(f"Number of negative samples per feature: {[len(f['neg_target']) for f in features]}")

#         # Collect all targets (positive and negative samples) maintaining order
#         all_targets = []
#         all_ne = []
#         src_indices = []  # Indices to map targets back to their sources

#         valid_contrasts = []
#         positive_contrasts = []

#         accumulate = 0
#         for i, feature in enumerate(features):
#             # Add positive samples first
#             valid_i = []
#             for pos_tgt, pos_ne in zip(feature["pos_target"], feature["pos_ne"]):
#                 all_targets.append(pos_tgt)
#                 all_ne.append(pos_ne)
#                 src_indices.append(i)  # Map each target to its source index

#                 if not all(x == 0 for x in pos_ne):
#                     valid_i.append(accumulate)
#                 accumulate += 1

#             # Add valid contrasts for positives
#             for idx1, idx2 in itertools.combinations(valid_i, 2):
#                 positive_contrasts.append((idx1, idx2))
#                 valid_contrasts.append((idx1, idx2))

#             # Then add negative samples
#             valid_j = []
#             for neg_tgt, neg_ne in zip(feature["neg_target"], feature["neg_ne"]):
#                 all_targets.append(neg_tgt)
#                 all_ne.append(neg_ne)
#                 src_indices.append(i)  # Map each target to its source index
#                 valid_j.append(accumulate)
#                 accumulate += 1

#             # Add valid contrasts for negatives
#             for idx1, idx2 in itertools.combinations(valid_j, 2):
#                 valid_contrasts.append((idx1, idx2))

#         # Now, for each target, prepare the input and labels
#         input_texts = []
#         labels_list = []
#         adjusted_ne = []

#         # Keep track of indices to remove samples that exceed max_length
#         valid_indices = []

#         for idx, (target_text, ne, src_idx) in enumerate(zip(all_targets, all_ne, src_indices)):
#             source_text = sources[src_idx]

#             # Truncate source and target texts to fit within max_length
#             # Allocate max_length between source and target
#             max_source_length = int(self.max_length * 0.6)  # Allocate 60% to source
#             max_target_length = self.max_length - max_source_length

#             # Tokenize source text with adjusted max_length
#             source_encoding = self.tokenizer(
#                 source_text,
#                 truncation=True,
#                 max_length=max_source_length,
#                 add_special_tokens=False,
#             )
#             source_len = len(source_encoding['input_ids'])
#             self.logger.debug(f"Source tokenized length: {source_len}")

#             # Tokenize target text with remaining length
#             target_encoding = self.tokenizer(
#                 target_text,
#                 truncation=True,
#                 max_length=max_target_length,
#                 add_special_tokens=False,
#             )
#             target_len = len(target_encoding['input_ids'])
#             self.logger.debug(f"Target tokenized length: {target_len}")

#             # Concatenate source and target input IDs
#             input_ids = source_encoding['input_ids'] + target_encoding['input_ids']
#             input_ids = input_ids[:self.max_length]  # Ensure total length does not exceed max_length

#             # If target tokens are missing after truncation, skip this sample
#             if target_len == 0:
#                 self.logger.warning(f"No target tokens left after truncation for sample {idx}. Skipping.")
#                 continue  # Skip to the next sample

#             # Create labels
#             labels = [-100] * source_len + target_encoding['input_ids']
#             labels = labels[:len(input_ids)]  # Ensure labels and input_ids are same length

#             # Adjust NE sequence to match target tokens
#             ne = ne[:target_len]
#             ne_padded = ne + [0] * (target_len - len(ne))

#             adjusted_ne_seq = [0] * source_len + ne_padded
#             adjusted_ne_seq = adjusted_ne_seq[:len(input_ids)]  # Ensure same length

#             # Add to lists
#             input_texts.append(self.tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))
#             labels_list.append(labels)
#             adjusted_ne.append(adjusted_ne_seq)
#             valid_indices.append(idx)

#         # After processing, adjust contrast matrices and src_indices to only include valid samples
#         if not valid_indices:
#             # If no valid samples remain, return an empty batch
#             return {}

#         # Filter src_indices based on valid_indices
#         src_indices = [src_indices[i] for i in valid_indices]

#         # Adjust contrast matrices
#         total_targets = len(valid_indices)
#         valid_contrast_matrix = torch.zeros((total_targets, total_targets), dtype=torch.bool)
#         positive_contrast_matrix = torch.zeros((total_targets, total_targets), dtype=torch.bool)

#         # Re-map indices
#         idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}
#         new_valid_contrasts = []
#         new_positive_contrasts = []

#         for idx1, idx2 in valid_contrasts:
#             if idx1 in idx_mapping and idx2 in idx_mapping:
#                 new_idx1 = idx_mapping[idx1]
#                 new_idx2 = idx_mapping[idx2]
#                 valid_contrast_matrix[new_idx1, new_idx2] = True
#                 valid_contrast_matrix[new_idx2, new_idx1] = True
#                 new_valid_contrasts.append((new_idx1, new_idx2))

#         for idx1, idx2 in positive_contrasts:
#             if idx1 in idx_mapping and idx2 in idx_mapping:
#                 new_idx1 = idx_mapping[idx1]
#                 new_idx2 = idx_mapping[idx2]
#                 positive_contrast_matrix[new_idx1, new_idx2] = True
#                 positive_contrast_matrix[new_idx2, new_idx1] = True
#                 new_positive_contrasts.append((new_idx1, new_idx2))

#         # Remove self-contrasts
#         for i in range(total_targets):
#             valid_contrast_matrix[i, i] = False
#             positive_contrast_matrix[i, i] = False

#         # Now tokenize the concatenated inputs with padding
#         batch_encoding = self.tokenizer(
#             input_texts,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors='pt',
#             add_special_tokens=False,
#         )

#         # Now we can update labels to match the padded input_ids
#         input_ids = batch_encoding['input_ids']
#         attention_mask = batch_encoding['attention_mask']

#         # Pad labels and adjusted_ne to match the padded input_ids
#         max_seq_len = input_ids.size(1)
#         padded_labels_list = []
#         padded_ne_list = []

#         for labels, ne_seq in zip(labels_list, adjusted_ne):
#             # Pad labels
#             if len(labels) < max_seq_len:
#                 labels += [-100] * (max_seq_len - len(labels))
#             else:
#                 labels = labels[:max_seq_len]
#             padded_labels_list.append(labels)

#             # Pad NE sequence
#             if len(ne_seq) < max_seq_len:
#                 ne_seq += [0] * (max_seq_len - len(ne_seq))
#             else:
#                 ne_seq = ne_seq[:max_seq_len]
#             padded_ne_list.append(ne_seq)

#         labels = torch.tensor(padded_labels_list, dtype=torch.long)
#         ne_tensor = torch.tensor(padded_ne_list, dtype=torch.long)

#         # Compute ntokens
#         ntokens = (labels != -100).sum().item()
          
#         # After collecting contrasts
#         self.logger.info(f"Number of valid pairs collected: {len(valid_contrasts)}")
#         self.logger.info(f"Number of positive pairs collected: {len(positive_contrasts)}")
        
#         # After filtering valid samples
#         self.logger.info(f"Number of valid samples after filtering: {len(valid_indices)}")
#         self.logger.info(f"Number of samples removed due to length: {len(features) - len(valid_indices)}")
        
#         # After remapping contrasts
#         self.logger.info(f"Number of valid pairs after remapping: {len(new_valid_contrasts)}")
#         self.logger.info(f"Number of positive pairs after remapping: {len(new_positive_contrasts)}")


#         assert input_ids.size(0) == attention_mask.size(0), \
#             f"Batch size mismatch: input_ids ({input_ids.size(0)}) vs attention_mask ({attention_mask.size(0)})"
#         assert input_ids.size(1) == attention_mask.size(1), \
#             f"Sequence length mismatch: input_ids ({input_ids.size(1)}) vs attention_mask ({attention_mask.size(1)})"
#         assert input_ids.size() == labels.size(), \
#             f"Shape mismatch: input_ids {input_ids.size()} vs labels {labels.size()}"
#         assert input_ids.size() == ne_tensor.size(), \
#             f"Shape mismatch: input_ids {input_ids.size()} vs ne_tensor {ne_tensor.size()}"

#         # Contrast matrix checks
#         assert valid_contrast_matrix.size() == (len(input_ids), len(input_ids)), \
#             f"Valid contrast matrix shape mismatch: {valid_contrast_matrix.size()} vs expected {(len(input_ids), len(input_ids))}"
#         assert positive_contrast_matrix.size() == (len(input_ids), len(input_ids)), \
#             f"Positive contrast matrix shape mismatch: {positive_contrast_matrix.size()} vs expected {(len(input_ids), len(input_ids))}"

#         # Source index checks
#         assert len(src_indices) == len(input_ids), \
#             f"Source indices length mismatch: {len(src_indices)} vs batch size {len(input_ids)}"
#         assert max(src_indices) < len(features), \
#             f"Invalid source index: max index {max(src_indices)} vs number of features {len(features)}"

#         # Value range checks
#         assert (attention_mask >= 0).all() and (attention_mask <= 1).all(), \
#             "Attention mask values must be 0 or 1"
#         assert (ne_tensor >= 0).all(), \
#             "Named entity tensor cannot have negative values"
#         assert ((labels == -100) | (labels >= 0)).all(), \
#             "Labels must be either -100 or non-negative"

#         # Mask alignment check
#         assert (labels == -100).sum() + ntokens == labels.numel(), \
#             f"Sum of ignored (-100) and counted tokens ({(labels == -100).sum() + ntokens}) should equal total elements ({labels.numel()})"

#         # Add assertions about contrasts
#         assert len(positive_contrasts) > 0, "No positive contrasts found in batch"
#         assert len(valid_contrasts) > 0, "No valid contrasts found in batch"
        
#         # Add assertion about sequence lengths
#         assert max_seq_len <= self.max_length, f"Sequence length {max_seq_len} exceeds max_length {self.max_length}"
        
#         # Check if we have enough samples for contrastive learning
#         assert len(valid_indices) >= 2, "Need at least 2 samples for contrastive learning"

#         # Debug logging
#         self.logger.info(f"Batch statistics:")
#         self.logger.info(f"- Batch size: {len(input_ids)}")
#         self.logger.info(f"- Sequence length: {input_ids.size(1)}")
#         self.logger.info(f"- Number of tokens (excluding padding): {ntokens}")
#         self.logger.info(f"- Number of valid contrasts: {valid_contrast_matrix.sum().item() // 2}")
#         self.logger.info(f"- Number of positive contrasts: {positive_contrast_matrix.sum().item() // 2}")

#         return {
#             "net_input": {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#             },
#             "target": labels,
#             "contrast_src_select_index": torch.tensor(src_indices, dtype=torch.long),
#             "valid_contrast": valid_contrast_matrix,
#             "positive_contrast": positive_contrast_matrix,
#             "contrast_ne": ne_tensor,
#             "ntokens": ntokens
#         }