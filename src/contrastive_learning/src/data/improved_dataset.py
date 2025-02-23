import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np
from src.data.debug_saver import DataLogger
import re

logger = logging.getLogger(__name__)
class SkipBatchException(Exception):
    pass

def sentence_tokenize(text: str) -> list[str]:
    # Splits on punctuation followed by whitespace
    return re.split(r'(?<=[.!?])\s+', text.strip())

def truncate_to_complete_sentences(text: str, tokenizer, max_length: int) -> str:
    sentences = sentence_tokenize(text)
    truncated = ""
    for sentence in sentences:
        candidate = truncated + (" " if truncated else "") + sentence
        tokens = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(tokens) <= max_length:
            truncated = candidate
        else:
            break
    return truncated

class FilteredContrastiveDataset(Dataset):
    """
    A dataset that:
    1) Loads source and target text files for positives and negatives.
    2) Looks up the .mapping.jsonl file (one JSON per line) to get
       `orig_idx` for each sample, instead of using the .other file.
    3) Filters out samples based on length constraints.
    4) Exposes each item by its new index, while referencing the old index
       for alignment of positives/negatives.
    """

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
        logger=None,
        min_target_length: int = 1
    ):
        """
        :param pos_data: Path to the 'pos' folder containing subfolders:
           'original', (possibly) 'back_translation', etc.
        :param neg_data: Path to the 'neg' folder containing subfolders with
           negative augmentations.
        :param neg_types: List of subfolder names for negative augmentations.
        :param model_name_or_path: Used to identify the subfolder for the tokeniser and also
           to parse the correct model name from "name_or_path".
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f"Initialising FilteredContrastiveDataset with split={split}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neg_samples = max_neg_samples
        self.min_target_length = min_target_length
        self.epoch = epoch - 1  # 0-based epoch
        self.neg_types = defaultdict(list)  # new: store augmentation type for each negative
        self.neg_types_all = neg_types  # for reference
        self.loaded_neg_types = set()

        try:
            # If model_name_or_path is e.g. "facebook/bart-large", we take "bart-large" as model_name
            model_name = model_name_or_path.split("/")[1]
        except IndexError:
            model_name = "gpt2"

        # 1) Load positive data
        self.original_pos_samples = defaultdict(list)
        self.original_pos_ne = defaultdict(list)
        indices_with_pos = set()

        # Load 'original' positives
        self._load_positives(
            base_path=os.path.join(pos_data, "original"),
            model_name=model_name,
            split=split,
            indices_with_pos=indices_with_pos,
            pos_samples=self.original_pos_samples,
            pos_ne=self.original_pos_ne
        )

        # Possibly load more positive augmentations if folder "back_translation" is present
        back_translation_path = os.path.join(pos_data, "back_translation")
        if os.path.exists(back_translation_path):
            self._load_positives(
                base_path=back_translation_path,
                model_name=model_name,
                split=split,
                indices_with_pos=indices_with_pos,
                pos_samples=self.original_pos_samples,
                pos_ne=self.original_pos_ne
            )

        # 2) Load negative data
        self.original_neg_samples = defaultdict(list)
        self.original_neg_ne = defaultdict(list)
        self.loaded_neg_types = set()
        indices_with_neg = set()
        if neg_types and neg_data:
            self._load_negatives(
                neg_data=neg_data,
                model_name=model_name,
                split=split,
                indices_with_neg=indices_with_neg,
                neg_samples=self.original_neg_samples,
                neg_ne=self.original_neg_ne,
                neg_types=neg_types
            )

        # 3) Identify the overlap (indices that have both pos and neg)
        valid_indices = indices_with_pos & indices_with_neg
        if not valid_indices:
            raise ValueError("No samples found with both positive and negative examples!")
        
        for idx in valid_indices:
            if idx in self.original_neg_samples:
                self._deduplicate_negatives(idx)

        # 4) Load sources from the 'original' positives folder
        source_file = os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.source")
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        with open(source_file, 'r', encoding='utf-8') as f:
            all_sources = [line.rstrip('\n') for line in f]

        # 5) Filter samples by length constraints
        self.filtered_indices = []
        for idx in valid_indices:
            if self._is_valid_sample(
                source=all_sources[idx],
                pos_ne=self.original_pos_ne[idx],
                pos_targets=self.original_pos_samples[idx],
                neg_targets=self.original_neg_samples[idx],
                neg_types=self.neg_types[idx],
            ):
                self.filtered_indices.append(idx)

        if not self.filtered_indices:
            raise ValueError("No valid samples after filtering by length constraints!")

        # 6) Create a new -> old index mapping
        self.idx_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(self.filtered_indices)}
        self.sources = [all_sources[idx] for idx in self.filtered_indices]

        # Group samples by unique source text
        aggregated_data = {}
        # 3 and 2 for llama
        max_neg_per_samples = 1000 #arbitrary big number
        max_pos_per_samples = 1000
        for idx in self.filtered_indices:
            source_text = all_sources[idx]
            if source_text not in aggregated_data:
                aggregated_data[source_text] = {
                    "source": source_text,
                    "pos_target": [],
                    "pos_ne": [],
                    "neg_target": [],
                    "neg_ne": [],
                    "augmentation_types": []
                }
            pos_target_ct = 0
            neg_target_ct = 0
            # Add positive targets (and their NE vectors), keeping only unique targets.
            for pos_tgt, pos_ne in zip(self.original_pos_samples[idx], self.original_pos_ne[idx]):
                if pos_tgt not in aggregated_data[source_text]["pos_target"]:
                    aggregated_data[source_text]["pos_target"].append(pos_tgt)
                    aggregated_data[source_text]["pos_ne"].append(pos_ne)
                    pos_target_ct += 1
                    if pos_target_ct > max_pos_per_samples:
                        break 
            # Add negative targets (and their NE vectors and augmentation types), keeping only unique targets.
            for neg_tgt, neg_ne, neg_aug in zip(self.original_neg_samples[idx],
                                                 self.original_neg_ne[idx],
                                                 self.neg_types[idx]):
                if neg_tgt not in aggregated_data[source_text]["neg_target"]:
                    aggregated_data[source_text]["neg_target"].append(neg_tgt)
                    aggregated_data[source_text]["neg_ne"].append(neg_ne)
                    aggregated_data[source_text]["augmentation_types"].append(neg_aug)
                    neg_target_ct += 1
                    if neg_target_ct > max_neg_per_samples:
                        break 

        # # Replace individual samples with the aggregated ones.
        self.samples = list(aggregated_data.values())
        self.logger.info(f"Aggregated dataset: {len(self.samples)} unique source samples.")
        
        # self.samples = list(valid_samples.values())
        self.logger.info(f"Filtered dataset contains {len(self.samples)} samples after excluding problematic ones.")

        # 7) Shuffle approach for negatives
        rng = np.random.RandomState(seed)
        self.neg_shuffle = {}
        for new_idx, old_idx in self.idx_mapping.items():
            neg_count = len(self.original_neg_samples[old_idx])
            if neg_count > 0:
                self.neg_shuffle[new_idx] = rng.permutation(neg_count)

        self.logger.info(
            f"FilteredContrastiveDataset: In split='{split}', found {len(self.filtered_indices)} valid samples "
            f"(pos={len(indices_with_pos)}, neg={len(indices_with_neg)})"
        )

    # --- Helper function to remove repeated sentences in a text ---
    def _remove_repeated_sentences(self, text: str) -> str:
        import re
        # Split text into sentences by recognising punctuation followed by whitespace.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return text
        dedup_sentences = [sentences[0]]
        for sentence in sentences[1:]:
            if sentence != dedup_sentences[-1]:
                dedup_sentences.append(sentence)
        return ' '.join(dedup_sentences)

    # --- Helper function to deduplicate negatives for a given sample index ---
    def _deduplicate_negatives(self, idx: int):
        original_targets = self.original_neg_samples[idx]
        original_ne = self.original_neg_ne[idx]
        original_aug = self.neg_types[idx]
        new_targets = []
        new_ne = []
        new_aug = []
        seen = set()
        for tgt, ne, aug in zip(original_targets, original_ne, original_aug):
            # Remove repeated sentences from the negative target.
            processed_tgt = self._remove_repeated_sentences(tgt)
            if processed_tgt not in seen:
                seen.add(processed_tgt)
                new_targets.append(processed_tgt)
                new_ne.append(ne)
                new_aug.append(aug)
        self.original_neg_samples[idx] = new_targets
        self.original_neg_ne[idx] = new_ne
        self.neg_types[idx] = new_aug

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching."""
        import re
        if not isinstance(text, str):
            return ""
        # Convert to lowercase and normalize whitespace
        text = text.lower().strip()
        text = ' '.join(text.split())
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        return text.strip()

    def _load_sources_and_create_mapping(self, source_file: str) -> Dict[str, int]:
        """Load source texts and create normalized text -> index mapping."""
        text_to_idx = {}
        with open(source_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                norm_text = self._normalize_text(line.strip())
                if norm_text:
                    text_to_idx[norm_text] = idx
        
        self.logger.info(f"Loaded {len(text_to_idx)} unique source texts")
        return text_to_idx

    def _load_positives(self, base_path, model_name, split, indices_with_pos, pos_samples, pos_ne):
        """
        Load positives with different strategies for original vs augmented data.
        """
        tgt_file = os.path.join(base_path, f"{model_name}/{split}/{split}.pos_target")
        ne_file = os.path.join(base_path, f"{model_name}/{split}/{split}.pos_ne")
        map_file = os.path.join(base_path, f"{model_name}/{split}/{split}.mapping.jsonl")
        source_file = os.path.join(base_path, f"{model_name}/{split}/{split}.source")

        if not all(os.path.exists(f) for f in [source_file, tgt_file, ne_file, map_file]):
            self.logger.warning(f"Missing files in {base_path}. Skipping.")
            return

        # Check if this is original or augmented data
        is_original = 'original' in base_path.split('/')

        if is_original:
            # For original data, use direct index mapping
            with open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
                open(ne_file, 'r', encoding='utf-8') as f_ne:
                
                for idx, (tgt_line, ne_line) in enumerate(zip(f_tgt, f_ne)):
                    pos_text = tgt_line.strip()
                    pos_ne_list = [int(x) for x in ne_line.strip().split()]
                    
                    pos_samples[idx].append(pos_text)
                    pos_ne[idx].append(pos_ne_list)
                    indices_with_pos.add(idx)
                    
            self.logger.debug(f"Loaded {len(indices_with_pos)} original positive samples")
        else:
            # For augmented data, use text matching
            orig_source_file = os.path.join(base_path, "..", "original", f"{model_name}/{split}/{split}.source")
            if not os.path.exists(orig_source_file):
                self.logger.error(f"Original source file not found: {orig_source_file}")
                return

            # Load original sources for matching
            source_mapping = {}
            with open(orig_source_file, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    norm_text = self._normalize_text(line.strip())
                    source_mapping[norm_text] = idx

            # Load current sources and match with originals
            with open(source_file, 'r', encoding='utf-8') as f_src, \
                open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
                open(ne_file, 'r', encoding='utf-8') as f_ne:
                
                matched = 0
                for src_line, tgt_line, ne_line in zip(f_src, f_tgt, f_ne):
                    norm_text = self._normalize_text(src_line.strip())
                    if norm_text in source_mapping:
                        orig_idx = source_mapping[norm_text]
                        
                        pos_text = tgt_line.strip()
                        pos_ne_list = [int(x) for x in ne_line.strip().split()]
                        
                        pos_samples[orig_idx].append(pos_text)
                        pos_ne[orig_idx].append(pos_ne_list)
                        indices_with_pos.add(orig_idx)
                        matched += 1

            self.logger.debug(f"Loaded {matched} augmented positive samples")

        total_samples = sum(len(samples) for samples in pos_samples.values())
        self.logger.info(f"Total positive samples loaded for {split}: {total_samples}")
        self.logger.info(f"Unique source indices: {len(indices_with_pos)}")
    
    
    def _load_negatives(self, neg_data, model_name, split, indices_with_neg, neg_samples, neg_ne, neg_types):
        """
        Enhanced negative loading with text-based matching.
        Expects each negative type directory to contain:
         - <split>.neg_target
         - <split>.neg_ne
         - <split>.mapping.jsonl
        The neg_types parameter is a list of folder names for negative augmentations.
        """
        orig_source_file = os.path.join(neg_data, "..", "pos/original", f"{model_name}/{split}/{split}.source")
        if not os.path.exists(orig_source_file):
            self.logger.error("Original source file not found for text matching")
            return

        source_mapping = self._load_sources_and_create_mapping(orig_source_file)

        for neg_type in neg_types:
            neg_type_dir = os.path.join(neg_data, neg_type)
            if not os.path.exists(neg_type_dir):
                self.logger.warning(f"Negative type directory not found: {neg_type_dir}")
                continue

            source_file = os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.source")
            tgt_file = os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.neg_target")
            ne_file = os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.neg_ne")
            map_file = os.path.join(neg_type_dir, f"{model_name}/{split}/{split}.mapping.jsonl")

            if not all(os.path.exists(f) for f in [source_file, tgt_file, ne_file, map_file]):
                self.logger.warning(f"Missing files in {neg_type_dir}, skipping.")
                continue

            self.logger.debug(f"Loading negatives from {neg_type_dir}")
            with open(source_file, 'r', encoding='utf-8') as f_src, \
                 open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
                 open(ne_file, 'r', encoding='utf-8') as f_ne, \
                 open(map_file, 'r', encoding='utf-8') as f_map:

                for src_line, tgt_line, ne_line, map_line in zip(f_src, f_tgt, f_ne, f_map):
                    try:
                        source_text = self._normalize_text(src_line.strip())
                        if not source_text:
                            continue
                        orig_idx = source_mapping.get(source_text, -1)
                        if orig_idx < 0:
                            continue
                        neg_text = tgt_line.strip()
                        neg_ne_list = [int(x) for x in ne_line.strip().split()]
                        # Try to parse mapping fields; if at least three fields are present, use the third; otherwise, default to folder name.
                        fields = map_line.strip().split('\t')
                        aug_type_field = neg_type
                        neg_samples[orig_idx].append(neg_text)
                        neg_ne[orig_idx].append(neg_ne_list)
                        # Store augmentation type for this negative sample in self.neg_types (a defaultdict)
                        self.neg_types[orig_idx].append(aug_type_field)
                        indices_with_neg.add(orig_idx)
                    except Exception as e:
                        self.logger.error(f"Error processing negative sample: {e}")
            self.loaded_neg_types.add(neg_type)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute the Jaccard similarity between two texts based on word tokens.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def _is_valid_sample(self, source: str, pos_ne, pos_targets: List[str], neg_targets: List[str], neg_types) -> bool:
        """
        Quick length-based filtering. If a sample fails,
        we skip it from the final set. In addition to length and token count checks,
        we also reject samples if any positive target is too similar to any negative target—
        unless the negative augmentation is toxic.
        """
        if len(pos_targets) < 2:
        # If we only have 1 (or 0) positive sub-samples, we won't have any positive pairs.
            self.logger.debug(f"Rejecting sample because it has fewer than 2 positives.")
            return False
        
        # Check that the first positive NE marker vector has enough active tokens.
        if sum(pos_ne[0][:int(self.max_length * 0.2)]) == 0:
            return False
        # Now check that the positive and negative targets are not too similar.
        # We use a Jaccard similarity measure (over word tokens) and reject if similarity exceeds threshold.
        similarity_threshold = 0.8  # adjust this threshold as needed
        for pos in pos_targets:
            for neg in neg_targets:
                sim = self._jaccard_similarity(pos, neg)
                if sim > similarity_threshold:
                    self.logger.debug(
                        f"Sample rejected due to high similarity (Jaccard {sim:.2f}) between pos and neg: '{pos}' vs. '{neg}'"
                    )
                    return False
        
        # If negatives are toxic, accept sample without checking for similarity.
        if 'toxic' or 'sys_lowcon' in neg_types:
            return True

        max_source_length = int(self.max_length * 0.8)
        source_tokens = self.tokenizer(
            source, truncation=True, max_length=max_source_length, add_special_tokens=False
        )
        if len(source_tokens["input_ids"]) == 0:
            return False

        # Check that each positive and negative target is of sufficient length.
        max_target_length = self.max_length - max_source_length
        for t in (pos_targets + neg_targets):
            tgt_tokens = self.tokenizer(
                t, truncation=True, max_length=max_target_length, add_special_tokens=False
            )
            if len(tgt_tokens["input_ids"]) < self.min_target_length:
                return False

        return True

    
    def __getitem__(self, index):
        """
        # Return a dictionary with source, up to 2 positive targets,
        # up to self.max_neg_samples negative targets, and augmentation types for negatives.
        # """
       # Get the sample directly from aggregated data
        sample = self.samples[index]
        
        return {
            "id": index,
            "source": sample["source"],
            "pos_target": sample["pos_target"],
            "neg_target": sample["neg_target"],
            "pos_ne": sample["pos_ne"],
            "neg_ne": sample["neg_ne"],
            "augmentation_types": sample["augmentation_types"]
        }

    def __len__(self) -> int:
        return len(self.sources)

    def set_epoch(self, epoch: int):
        self.epoch = epoch - 1

def validate_dataset_with_collator_logic(dataset, logger=None):
    """
    Validates a FilteredContrastiveDataset by simulating collator logic.
    Returns indices of valid samples that should be kept.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    valid_indices = []
    logger.info(f"Validating {len(dataset)} samples...")
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            # Core validation: must have 2+ positive targets after aggregation
            pos_targets = sample.get("pos_target", [])
            if len(pos_targets) < 2:
                logger.debug(f"Sample {idx}: Rejected - has only {len(pos_targets)} positive targets")
                continue
                
            # Additional collator-specific validations can be added here
            # For example: checking NE masks align, verifying target lengths, etc.
            
            valid_indices.append(idx)
            
        except Exception as e:
            logger.debug(f"Sample {idx}: Failed validation with error: {str(e)}")
            continue
    
    logger.info(f"Found {len(valid_indices)} valid samples out of {len(dataset)} total")
    return valid_indices

class ValidatedContrastiveDataset(Dataset):
    """
    A wrapper dataset that only includes samples that pass collator validation.
    """
    def __init__(self, base_dataset, logger=None):
        self.base_dataset = base_dataset
        self.logger = logger or logging.getLogger(__name__)
        
        # Get valid indices
        self.valid_indices = validate_dataset_with_collator_logic(
            base_dataset, 
            logger=self.logger
        )
        
        if not self.valid_indices:
            raise ValueError("No valid samples found after collator validation!")
            
        self.logger.info(f"ValidatedContrastiveDataset contains {len(self.valid_indices)} samples")
    
    def __getitem__(self, idx):
        return self.base_dataset[self.valid_indices[idx]]
    
    def __len__(self):
        return len(self.valid_indices)
    
    def set_epoch(self, epoch):
        # Pass through to base dataset if it has this method
        if hasattr(self.base_dataset, 'set_epoch'):
            self.base_dataset.set_epoch(epoch)


class ConsistentContrastiveCollator:
    """
    A collator that closely follows the original Fairseq-style contrastive logic,
    but adapts it for a decoder-only model. Each sample may contain:
      - source (text)
      - pos_target (list of positive target tensors)
      - neg_target (list of negative target tensors)
      - pos_ne (list of positive NE masks)
      - neg_ne (list of negative NE masks)

    The output batch has:
      - net_input.input_ids: stacked source+target tokens for each sub-sample (pos or neg)
      - net_input.attention_mask: attention mask for each sub-sample
      - labels: same size as input_ids, with the source portion usually set to -100 (so it won't contribute to loss)
      - contrast_src_select_index: indicates which original sample each sub-sample belongs to
      - contrast_ne: NE mask aligned with input_ids
      - valid_contrast: adjacency matrix indicating sub-samples from the same source
      - positive_contrast: adjacency matrix indicating sub-samples that are both positive from the same source
      - ce_pos: indices of the final positive sub-sample per source (matching original code)
      - ntokens: total number of non-padding tokens in the batch (for logging or similar)
    """

    def __init__(
        self,
        tokenizer,
        logger,
        data_logger=None,
        max_length=256,
        pad_to_multiple_of=1,
        left_pad_source=False,
        left_pad_target=False,
        input_feeding=True,
        source_mask_in_labels=True,
        add_bos_token=False
    ):
        """
        :param tokenizer: A huggingface/transformers tokenizer
        :param max_length: Maximum length of the combined source+target sequence
        :param pad_to_multiple_of: If > 1, pad sequences to a multiple of this number
        :param left_pad_source: Legacy param; for a decoder-only model it usually remains False
        :param left_pad_target: Same as above; typically False for a decoder-only approach
        :param input_feeding: Mirrors the old setting where targets shift for teacher-forcing
        :param source_mask_in_labels: If True, masks the source portion of input_ids with -100
                                      so that it doesn't affect the language modelling loss
        :param add_bos_token: If True, optionally add a BOS token at the start of the combined sequence
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.input_feeding = input_feeding
        self.source_mask_in_labels = source_mask_in_labels
        self.add_bos_token = add_bos_token
        self.data_logger = data_logger
        self.logger = logger

        if tokenizer.pad_token_id is None:
            # In some cases (GPT2-like), we may need to define the pad token
            self.pad_id = tokenizer.eos_token_id
        else:
            self.pad_id = tokenizer.pad_token_id

        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else self.eos_id

    def __call__(self, samples):
        """
        Merges a list of samples into a batch. Each sample may contain multiple positive
        and negative sub-samples. We collect them all, build adjacency matrices, and
        stack them for training.
        """
        if len(samples) == 0:
            print("No samples given to collator!!!, Returning dummy")
            dummy_input_ids = torch.full((1, self.max_length), self.tokenizer.pad_token_id, dtype=torch.long)
            dummy_attention_mask = torch.ones((1, self.max_length), dtype=torch.long)
            dummy_labels = torch.full((1, self.max_length), -100, dtype=torch.long)
            dummy_index = torch.tensor([0], dtype=torch.long)
            batch = {
                    "dummy": True,  # flag so you know to skip this batch
                    "id": dummy_index,
                    "nsentences": 1,
                    "ntokens": 0,
                    "net_input": {
                        "input_ids": dummy_input_ids,
                        "attention_mask": dummy_attention_mask,
                    },
                    "target": dummy_labels,
                    "contrast_src_select_index": dummy_index,
                    "contrast_ne": torch.zeros((1, self.max_length), dtype=torch.long),
                    "valid_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "positive_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "ce_pos": dummy_index
                }
            return batch
        
        samples = [s for s in samples if len(s.get("pos_target", [])) >= 2]
        if not samples:
            # print("ERROR HERE")
            # return None
            # Create empty tensors using self.max_length for sequence length.
            dummy_input_ids = torch.full((1, self.max_length), self.tokenizer.pad_token_id, dtype=torch.long)
            dummy_attention_mask = torch.ones((1, self.max_length), dtype=torch.long)
            dummy_labels = torch.full((1, self.max_length), -100, dtype=torch.long)
            dummy_index = torch.tensor([0], dtype=torch.long)
            batch = {
                    "dummy": True,  # flag so you know to skip this batch
                    "id": dummy_index,
                    "nsentences": 1,
                    "ntokens": 0,
                    "net_input": {
                        "input_ids": dummy_input_ids,
                        "attention_mask": dummy_attention_mask,
                    },
                    "target": dummy_labels,
                    "contrast_src_select_index": dummy_index,
                    "contrast_ne": torch.zeros((1, self.max_length), dtype=torch.long),
                    "valid_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "positive_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "ce_pos": dummy_index
                }
            print("Returning dummy because of pos_traget")
            return batch
    
        # We'll flatten out all positives and negatives
        flat_input_ids = []
        flat_attention_masks = []
        flat_labels = []
        flat_ne_masks = []
        contrast_src_select_index = []  # which sample each sub-sample came from
        is_pos_list = []               # track which sub-sample is positive vs negative
        ce_pos = []                    # track the last positive sub-sample per original sample

        sub_counter = 0  # tracks sub-samples across the entire batch

        accumulate_count = 0
        sample_ids = [s["id"] for s in samples]  # for logging if desired

        # Iterate over each sample
        for sample_idx, s in enumerate(samples):
            pos_list = s.get("pos_target", [])
            # logger.info(f"Sample {sample_idx} has {len(pos_list)} positives before processing.")
            pos_ne_list = s.get("pos_ne", [])
            neg_list = s.get("neg_target", [])
            neg_ne_list = s.get("neg_ne", [])

            # We expect len(pos_list) == len(pos_ne_list), etc.
            # First, handle positives
            count_for_this_sample = 0
            num_pos_this_sample = 0
            for p_idx, p_text in enumerate(pos_list):
                num_pos_this_sample +=1
            
            # to avoid pos ratio 0 issue in some subsamples
            # if num_pos_this_sample < 2:
            #     raise 

            for p_idx, p_text in enumerate(pos_list):
                p_ne = pos_ne_list[p_idx]
                # Convert to sub-sample
                out_dict = self._make_subsample(s["source"], p_text, p_ne, is_positive=True)
                flat_input_ids.append(out_dict["input_ids"])
                flat_attention_masks.append(out_dict["attention_mask"])
                flat_labels.append(out_dict["labels"])
                flat_ne_masks.append(out_dict["ne"])
                contrast_src_select_index.append(sample_idx)
                is_pos_list.append(True)
                sub_counter += 1
                count_for_this_sample +=1

            # Now handle negatives
            for n_idx, n_text in enumerate(neg_list):
                n_ne = neg_ne_list[n_idx]
                out_dict = self._make_subsample(s["source"], n_text, n_ne, is_positive=False)
                flat_input_ids.append(out_dict["input_ids"])
                flat_attention_masks.append(out_dict["attention_mask"])
                flat_labels.append(out_dict["labels"])
                flat_ne_masks.append(out_dict["ne"])
                contrast_src_select_index.append(sample_idx)
                is_pos_list.append(False)
                sub_counter += 1
                count_for_this_sample +=1

            print(f"Sample {sample_idx} => appended {count_for_this_sample} sub-samples, {num_pos_this_sample} marked positive.")
            # Original code's cross-entropy tracking: the last positive index
            # for this sample. They did: accumulate_count += num_pos; ce_pos.append(accumulate_count - 1); accumulate_count += num_neg
            num_pos = len(pos_list)
            num_neg = len(neg_list)
            accumulate_count += num_pos
            if num_pos > 0:
                ce_pos.append(accumulate_count - 1)  # index of the last positive
            else:
                # in some edge case if a sample had no positives, skip
                # but typically your dataset ensures at least 1 pos target
                ce_pos.append(-1)
            accumulate_count += num_neg

        if sub_counter == 0:
            # this is a possible raise condition if no sub-samples were created
            batch = {
                    "dummy": True,  # flag so you know to skip this batch
                    "id": dummy_index,
                    "nsentences": 1,
                    "ntokens": 0,
                    "net_input": {
                        "input_ids": dummy_input_ids,
                        "attention_mask": dummy_attention_mask,
                    },
                    "target": dummy_labels,
                    "contrast_src_select_index": dummy_index,
                    "contrast_ne": torch.zeros((1, self.max_length), dtype=torch.long),
                    "valid_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "positive_contrast": torch.zeros((1, 1), dtype=torch.bool),
                    "ce_pos": dummy_index
                }
            print("Returning an empty dummy batch, because sub-counter")
            return batch

        # Now we stack everything
        input_ids_tensor = torch.stack(flat_input_ids, dim=0)
        attention_mask_tensor = torch.stack(flat_attention_masks, dim=0)
        labels_tensor = torch.stack(flat_labels, dim=0)
        ne_tensor = torch.stack(flat_ne_masks, dim=0)

        # print("All input ids: \n", input_ids_tensor)
        # print("All attention ids: \n", attention_mask_tensor)
        # print("All labels ids: \n", labels_tensor)

        # Build valid_contrast and positive_contrast adjacency matrices
        # For all sub-samples belonging to the same sample, valid_contrast=True
        # For sub-samples that are both positive from that sample, positive_contrast=True
        bs = input_ids_tensor.size(0)
        valid_contrast = torch.zeros(bs, bs, dtype=torch.bool)
        positive_contrast = torch.zeros(bs, bs, dtype=torch.bool)

        # Group sub-samples by sample_idx
        sample_to_subindices = defaultdict(list)
        for sub_idx, s_idx in enumerate(contrast_src_select_index):
            sample_to_subindices[s_idx].append(sub_idx)

        # Fill in adjacency
        for s_idx, sub_indices in sample_to_subindices.items():
            # Mark them all valid_contrast within the same sample
            for i in range(len(sub_indices)):
                for j in range(i + 1, len(sub_indices)):
                    ii = sub_indices[i]
                    jj = sub_indices[j]
                    valid_contrast[ii, jj] = True
                    valid_contrast[jj, ii] = True

            # Mark positives as positive_contrast
            pos_only = [x for x in sub_indices if is_pos_list[x]]
            for i in range(len(pos_only)):
                for j in range(i + 1, len(pos_only)):
                    ii = pos_only[i]
                    jj = pos_only[j]
                    positive_contrast[ii, jj] = True
                    positive_contrast[jj, ii] = True

        # Remove diagonal self-contrast
        diag_idx = range(bs)
        valid_contrast[diag_idx, diag_idx] = False
        positive_contrast[diag_idx, diag_idx] = False
        print("positive_contrast sum =", positive_contrast.sum().item())

        # Count tokens
        ntokens = (labels_tensor != -100).sum().item()

        # Final batch structure
        batch = {
            "id": torch.LongTensor(sample_ids),
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor
            },
            "target": labels_tensor,
            "contrast_src_select_index": torch.tensor(contrast_src_select_index, dtype=torch.long),
            "contrast_ne": ne_tensor,  # NE mask aligned with each sub-sample
            "valid_contrast": valid_contrast,
            "positive_contrast": positive_contrast,
            "ce_pos": torch.tensor(ce_pos, dtype=torch.long) if len(ce_pos) == len(samples) else None
        }

        if self.data_logger:
            self.data_logger.save_batch(batch, samples)

        return batch
    
    def _make_subsample(self, source_text, target_text, target_ne, is_positive=True):
        # 428 works with gpt2
        # 256 was for phi2 and llama2
        source_text = truncate_to_complete_sentences(source_text, self.tokenizer, max_length=216)
        src_encoding = self.tokenizer(
            source_text,
            add_special_tokens=False,
            truncation=True,
            max_length=428,
        )
        src_ids = src_encoding["input_ids"]

        tgt_encoding = self.tokenizer(
            target_text,
            add_special_tokens=False,
            truncation=True,
            max_length=84,
        )
        tgt_ids = tgt_encoding["input_ids"]

        # Align the NE mask length if needed
        if len(tgt_ids) != len(target_ne):
            min_len = min(len(tgt_ids), len(target_ne))
            tgt_ids = tgt_ids[:min_len]
            target_ne = target_ne[:min_len]

        combined_ids = []
        combined_ne = []
        if self.add_bos_token:
            combined_ids.append(self.bos_id)
            combined_ne.append(0)

        # Source portion
        src_start = len(combined_ids)
        for sid in src_ids:
            combined_ids.append(sid)
            combined_ne.append(0)
        src_end = len(combined_ids)

        # Target portion
        tgt_start = src_end
        for i, tid in enumerate(tgt_ids):
            combined_ids.append(tid)
            combined_ne.append(target_ne[i] if i < len(target_ne) else 0)
        tgt_end = len(combined_ids)

        # Truncate if too long
        if len(combined_ids) > self.max_length:
            combined_ids = combined_ids[:self.max_length]
            combined_ne = combined_ne[:self.max_length]
            tgt_end = min(tgt_end, self.max_length)

        # Create labels: shift target by one position
        labels = [-100] * len(combined_ids)

        # Example: for positions [tgt_start .. tgt_end-2], label[i] = combined_ids[i+1]
        for i in range(tgt_start, tgt_end - 1):
            labels[i] = combined_ids[i + 1]

        # The last token of the sequence won't predict anything (or could predict EOS),
        # so we leave it at -100.

        # Source tokens remain masked out at -100, so they don't contribute to LM loss.

        # Pad if needed
        pad_len = self.max_length - len(combined_ids)
        if pad_len > 0:
            combined_ids += [self.pad_id] * pad_len
            combined_ne += [0] * pad_len
            labels += [-100] * pad_len

        attention_mask = [1 if x != self.pad_id else 0 for x in combined_ids]

        return {
            "input_ids": torch.tensor(combined_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "ne": torch.tensor(combined_ne, dtype=torch.long)
        }

def prepare_filtered_data_for_training(
    data_args,
    training_args,
    model_name_or_path,
    tokenizer,
    accelerator,
    logger,
    max_eval_samples: Optional[int] = 32,
    seed: int = None,
):
    """
    Example function that:
      1) Creates FilteredContrastiveDataset for train split
      2) Optionally creates FilteredContrastiveDataset for validation split
         or splits the train dataset into train/val
      3) Builds a collator with the chosen left/right padding and returns them
    """
    logger.info("Loading and filtering dataset...")
    padding_side = "left" if "gpt" in model_name_or_path else "right"
    logger.info(f"Padding side {padding_side}")
    pos_data = os.path.join(data_args.data_path, data_args.pos_data_dir)
    neg_data = os.path.join(data_args.data_path, data_args.neg_data_dir)

    # Create training dataset
    train_dataset = FilteredContrastiveDataset(
        data_path=data_args.data_path,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        split="train",
        max_length=data_args.max_length,
        pos_data=pos_data,
        neg_data=neg_data,
        neg_types=data_args.neg_types,
        max_neg_samples=data_args.max_neg_samples,
        seed=seed,
        epoch=1,
        logger=logger,
        min_target_length=1
    )
    logger.info(f"Training dataset size = {len(train_dataset)}")
    train_dataset = ValidatedContrastiveDataset(train_dataset, logger=logger)
    logger.info(f"Training dataset size = {len(train_dataset)}")
    # Create separate validation dataset if needed
    if data_args.validation_split and data_args.validation_split != "train":
        eval_dataset = FilteredContrastiveDataset(
            data_path=data_args.data_path,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            split=data_args.validation_split,
            max_length=data_args.max_length,
            pos_data=pos_data,
            neg_data=neg_data,
            neg_types=data_args.neg_types,
            max_neg_samples=data_args.max_neg_samples,
            seed=seed,
            epoch=1,
            logger=logger,
            min_target_length=1
        )
        eval_dataset = ValidatedContrastiveDataset(eval_dataset, logger=logger)
        val_size_before = len(eval_dataset)
        if max_eval_samples and max_eval_samples < val_size_before:
            eval_dataset = Subset(eval_dataset, range(max_eval_samples))
            logger.info(f"Reduced validation dataset to {max_eval_samples} samples out of {val_size_before}")
    else:
        # Otherwise, do a random split of the train dataset
        if max_eval_samples >= len(train_dataset):
            raise ValueError("max_eval_samples is too large for the training set!")
        val_size = max_eval_samples
        train_size = len(train_dataset) - val_size
        train_dataset, eval_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        logger.info(f"Split train dataset into train={train_size}, validation={val_size}.")

    data_logger = DataLogger(output_dir=training_args.output_dir)

    # Create the collator with left or right padding
    collator = ConsistentContrastiveCollator(
        logger=logger,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        left_pad_source=True,
        left_pad_target=False,
        pad_to_multiple_of=1,
        input_feeding=True,
        add_bos_token=True,
        # samples_per_batch=training_args.per_device_train_batch_size,  # or any desired batch size
        # padding_side=padding_side,
        data_logger=data_logger,
    )

    # Quick debug: try a single batch
    loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=collator, num_workers=0)
    for i, batch in enumerate(loader):
        logger.info(f"Inspected training batch {i} => shape: {batch['net_input']['input_ids'].shape}")
        break

    logger.info(f"Final train dataset size: {len(train_dataset)}")
    logger.info(f"Final eval dataset size: {len(eval_dataset)}")

    logger.info("Inspecting dataset samples...")
    # inspect_data_sample(train_dataset, collator, tokenizer, num_samples=10)

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
        print("Aug type:", sample["augmentation_types"],"\n")
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
    loader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=True)
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
