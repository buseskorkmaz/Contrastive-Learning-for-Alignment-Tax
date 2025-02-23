import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from transformers import PreTrainedTokenizer
import numpy as np

# If your code relies on this, keep it:
from src.data.debug_saver import DataLogger
# just a backup-should be deleted
##############################################################################
#                          FilteredContrastiveDataset                        #
##############################################################################

class FilteredContrastiveDataset(Dataset):
    """
    A dataset that:
    1) Loads source and target text files for positives and negatives.
    2) Looks up the .mapping.jsonl file (one JSON per line) to get
       `orig_idx` for each sample, instead of using the .other file.
    3) Filters out samples based on length constraints.
    4) Exposes each item by its new index, while referencing the old index
       for alignment of positives/negatives.

    Each dataset item is "one source + multiple positive expansions + multiple negative expansions."
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
        :param pos_data: Path to the 'pos' folder containing subfolders like 'original', 'back_translation', etc.
        :param neg_data: Path to the 'neg' folder containing subfolders with negative augmentations.
        :param neg_types: List of subfolder names for negative augmentations.
        :param model_name_or_path: Used to identify the subfolder for the tokenizer and parse e.g. "bart-large"
        """
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug(f"Initialising FilteredContrastiveDataset with split={split}")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neg_samples = max_neg_samples
        self.min_target_length = min_target_length
        self.epoch = epoch - 1  # 0-based epoch
        self.neg_types = defaultdict(list)  # store augmentation type for each negative
        self.neg_types_all = neg_types
        self.loaded_neg_types = set()

        try:
            # If model_name_or_path is e.g. "facebook/bart-large", take "bart-large" as model_name
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

        # Possibly load more positive augmentations if "back_translation" is present
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

        # 4) Load sources from the 'original' positives folder
        source_file = os.path.join(pos_data, "original", f"{model_name}/{split}/{split}.source")
        if not os.path.exists(source_file):
            raise FileNotFoundError(f"Source file not found: {source_file}")

        with open(source_file, 'r', encoding='utf-8') as f:
            all_sources = [line.rstrip('\n') for line in f]

        # 5) Filter out invalid samples
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

        # 6) new -> old index mapping
        self.idx_mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(self.filtered_indices)}
        self.sources = [all_sources[idx] for idx in self.filtered_indices]

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

    def _normalize_text(self, text: str) -> str:
        """Simple text normalisation for matching."""
        import re
        if not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = ' '.join(text.split())
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        return text.strip()

    def _load_sources_and_create_mapping(self, source_file: str) -> Dict[str, int]:
        """Load source texts and map normalised text -> index for matching."""
        text_to_idx = {}
        with open(source_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                norm_text = self._normalize_text(line.strip())
                if norm_text:
                    text_to_idx[norm_text] = idx
        self.logger.info(f"Loaded {len(text_to_idx)} unique source texts for matching")
        return text_to_idx

    def _load_positives(self, base_path, model_name, split, indices_with_pos, pos_samples, pos_ne):
        """
        Load positives from e.g.:
           {split}.pos_target
           {split}.pos_ne
           {split}.mapping.jsonl
        If it's 'original', we do direct index mapping; otherwise we match text.
        """
        tgt_file = os.path.join(base_path, f"{model_name}/{split}/{split}.pos_target")
        ne_file = os.path.join(base_path, f"{model_name}/{split}/{split}.pos_ne")
        map_file = os.path.join(base_path, f"{model_name}/{split}/{split}.mapping.jsonl")
        source_file = os.path.join(base_path, f"{model_name}/{split}/{split}.source")

        if not all(os.path.exists(f) for f in [source_file, tgt_file, ne_file, map_file]):
            self.logger.warning(f"Missing files in {base_path}. Skipping.")
            return

        is_original = 'original' in base_path.split('/')

        if is_original:
            # For original data, direct index usage
            with open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
                 open(ne_file, 'r', encoding='utf-8') as f_ne:
                for idx, (tgt_line, ne_line) in enumerate(zip(f_tgt, f_ne)):
                    pos_text = tgt_line.strip()
                    pos_ne_list = [int(x) for x in ne_line.strip().split()]
                    
                    pos_samples[idx].append(pos_text)
                    pos_ne[idx].append(pos_ne_list)
                    indices_with_pos.add(idx)
            self.logger.debug(f"Loaded up to {len(indices_with_pos)} original positive samples")
        else:
            # For augmented data, match current src lines to original src
            orig_source_file = os.path.join(base_path, "..", "original", f"{model_name}/{split}/{split}.source")
            if not os.path.exists(orig_source_file):
                self.logger.error(f"Original source file not found: {orig_source_file}")
                return

            # Build mapping from normalized text -> original index
            source_mapping = {}
            with open(orig_source_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    norm_text = self._normalize_text(line.strip())
                    source_mapping[norm_text] = i

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

        total_samples = sum(len(ps) for ps in pos_samples.values())
        self.logger.info(f"Total positive samples loaded for {split}: {total_samples}")
        self.logger.info(f"Unique source indices so far: {len(indices_with_pos)}")

    def _load_negatives(self, neg_data, model_name, split, indices_with_neg, neg_samples, neg_ne, neg_types):
        """
        Load negative examples similarly, from subfolders. Expects:
           {split}.neg_target
           {split}.neg_ne
           {split}.mapping.jsonl
        """
        orig_source_file = os.path.join(neg_data, "..", "pos/original", f"{model_name}/{split}/{split}.source")
        if not os.path.exists(orig_source_file):
            self.logger.error("Original source file not found for text matching")
            return

        # Build text->idx mapping for the original source
        source_mapping = self._load_sources_and_create_mapping(orig_source_file)

        for ntype in neg_types:
            ntype_dir = os.path.join(neg_data, ntype)
            if not os.path.exists(ntype_dir):
                self.logger.warning(f"Negative type directory not found: {ntype_dir}")
                continue

            source_file = os.path.join(ntype_dir, f"{model_name}/{split}/{split}.source")
            tgt_file    = os.path.join(ntype_dir, f"{model_name}/{split}/{split}.neg_target")
            ne_file     = os.path.join(ntype_dir, f"{model_name}/{split}/{split}.neg_ne")
            map_file    = os.path.join(ntype_dir, f"{model_name}/{split}/{split}.mapping.jsonl")

            if not all(os.path.exists(f) for f in [source_file, tgt_file, ne_file, map_file]):
                self.logger.warning(f"Missing negative files in {ntype_dir}; skipping.")
                continue

            self.logger.debug(f"Loading negatives from {ntype_dir}")
            with open(source_file, 'r', encoding='utf-8') as f_src, \
                 open(tgt_file, 'r', encoding='utf-8') as f_tgt, \
                 open(ne_file, 'r', encoding='utf-8') as f_ne, \
                 open(map_file, 'r', encoding='utf-8') as f_map:

                for src_line, tgt_line, ne_line, map_line in zip(f_src, f_tgt, f_ne, f_map):
                    try:
                        norm_text = self._normalize_text(src_line.strip())
                        if not norm_text:
                            continue
                        orig_idx = source_mapping.get(norm_text, -1)
                        if orig_idx < 0:
                            continue
                        neg_text = tgt_line.strip()
                        neg_ne_list = [int(x) for x in ne_line.strip().split()]

                        self.neg_types[orig_idx].append(ntype)
                        neg_samples[orig_idx].append(neg_text)
                        neg_ne[orig_idx].append(neg_ne_list)
                        indices_with_neg.add(orig_idx)
                    except Exception as e:
                        self.logger.error(f"Error processing negative sample: {e}")

            self.loaded_neg_types.add(ntype)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Compute the Jaccard similarity over whitespace-split tokens.
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))

    def _is_valid_sample(self, source: str, pos_ne, pos_targets: List[str], neg_targets: List[str], neg_types) -> bool:
        """
        Quick length-based filtering, plus a Jaccard check that positives and negatives
        aren't overly similar, unless the neg type is 'toxic' or 'sys_lowcon'.
        """
        # e.g., ensure at least one NE in the first pos
        if sum(pos_ne[0][:int(self.max_length * 0.2)]) == 0:
            return False

        # Jaccard for pos vs neg
        similarity_threshold = 0.8
        for pt in pos_targets:
            for nt in neg_targets:
                sim = self._jaccard_similarity(pt, nt)
                if sim > similarity_threshold:
                    self.logger.debug(
                        f"Rejected sample (pos vs neg too similar, {sim:.2f}): '{pt[:50]}...' vs '{nt[:50]}...'"
                    )
                    return False

        # if 'toxic' or 'sys_lowcon' in neg_types, skip the check, but we already did it
        # so basically we always "accept" if that was the reason

        # Check source is not empty
        max_source_length = int(self.max_length * 0.8)
        source_tok = self.tokenizer(
            source,
            truncation=True,
            max_length=max_source_length,
            add_special_tokens=False
        )
        if len(source_tok["input_ids"]) == 0:
            return False

        # Also check each target not < min_target_length
        for t in (pos_targets + neg_targets):
            tgt_tok = self.tokenizer(
                t,
                truncation=True,
                max_length=self.max_length - max_source_length,
                add_special_tokens=False
            )
            if len(tgt_tok["input_ids"]) < self.min_target_length:
                return False
        return True

    def __getitem__(self, new_idx):
        """
        Return a dictionary with:
        "id": the new_idx (sample ID in filtered set)
        "source": a list of token IDs for the source text
        "pos_target": a list of token-ID lists, each from a positive text
        "pos_ne": a list of NE arrays for those positives
        "neg_target": a list of token-ID lists, each from a negative text
        "neg_ne": a list of NE arrays for those negatives
        "augmentation_types": the negative augmentation types for each negative
        """
        old_idx = self.idx_mapping[new_idx]

        # 1) Convert the raw source text to token IDs.
        source_str = self.sources[new_idx]
        source_enc = self.tokenizer(
            source_str,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        source_ids = source_enc["input_ids"]  # A Python list of int token IDs

        # 2) Grab up to 2 positive texts, then tokenize each
        pos_texts = self.original_pos_samples[old_idx][:2]
        pos_ne = self.original_pos_ne[old_idx][:2]

        tokenised_pos_targets = []
        for pt_str in pos_texts:
            pt_enc = self.tokenizer(
                pt_str,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            )
            tokenised_pos_targets.append(pt_enc["input_ids"])

        # 3) Handle the negative texts and NE arrays (plus augmentation types)
        if old_idx in self.original_neg_samples:
            if len(self.original_neg_samples[old_idx]) <= self.max_neg_samples:
                neg_texts = self.original_neg_samples[old_idx]
                neg_ne_arrays = self.original_neg_ne[old_idx]
                neg_aug = self.neg_types[old_idx]
            else:
                start_idx = self.epoch * self.max_neg_samples
                shuffle = self.neg_shuffle[new_idx]
                indices = [
                    shuffle[i % len(self.original_neg_samples[old_idx])]
                    for i in range(start_idx, start_idx + self.max_neg_samples)
                ]
                neg_texts = [self.original_neg_samples[old_idx][i] for i in indices]
                neg_ne_arrays = [self.original_neg_ne[old_idx][i] for i in indices]
                neg_aug = [self.neg_types[old_idx][i] for i in indices]
        else:
            neg_texts = []
            neg_ne_arrays = []
            neg_aug = []

        # 4) Tokenise each negative text
        tokenised_neg_targets = []
        for nt_str in neg_texts:
            nt_enc = self.tokenizer(
                nt_str,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            )
            tokenised_neg_targets.append(nt_enc["input_ids"])

        return {
            "id": new_idx,
            # The token IDs for the source text
            "source": source_ids,

            # The token IDs for each positive text
            "pos_target": tokenised_pos_targets,
            "pos_ne": pos_ne,

            # The token IDs for each negative text
            "neg_target": tokenised_neg_targets,
            "neg_ne": neg_ne_arrays,

            # The negative augmentation types
            "augmentation_types": neg_aug
        }

    def __len__(self):
        return len(self.filtered_indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch - 1


##############################################################################
#                         The Collator (Old BART-Style)                      #
##############################################################################

def contrastive_collate_gpt(
    samples,
    tokenizer,
    left_pad_source: bool = True,
    left_pad_target: bool = False,
    include_ne_masks: bool = True
):
    """
    Collation function that arranges data in the style of old contrastive BART,
    but for a decoder-only GPT, *including* a 1-token shift for cross-entropy:
    
      - 'contrast_prev_output_tokens' is the decoder input, forcibly shifted right.
      - 'contrast_target' is the original unshifted sequence.

    Steps:
      1) Collate the source tokens (one row per sample).
      2) Flatten all positives + negatives across the batch => a big list of target sequences.
      3) For each target sequence, build:
         (a) 'contrast_target' = the original tokens, padded
         (b) 'contrast_prev_output_tokens' = that same sequence, but shifted right by 1 token
             and typically the first token is BOS or pad.
      4) Build your valid_contrast, positive_contrast arrays.
      5) Return a dictionary with all needed fields.

    Args:
      samples: dataset items, each of which has:
        - "source": token IDs for the source
        - "pos_target"/"neg_target": lists of token-ID lists
        - optional "pos_ne"/"neg_ne" if using named-entity arrays
      tokenizer: a PreTrainedTokenizer with .pad_token_id and possibly .bos_token_id
      left_pad_source: whether to left-pad the source array
      left_pad_target: whether to left-pad the target arrays
      include_ne_masks: whether to collate pos_ne/neg_ne
    """
    if len(samples) == 0:
        return {}

    pad_id = tokenizer.pad_token_id
    # If your GPT has a BOS token, use it. If it doesn't, you can just reuse pad or a special token.
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else pad_id

    def collate_1d(list_of_token_lists, pad_id, left_pad=False):
        """
        Takes e.g. [[101,102],[101,102,103],[150]] and returns a [batch_size, max_len] LongTensor,
        with optional left-padding.
        """
        if not list_of_token_lists:
            return torch.empty(0, dtype=torch.long)
        max_len = max(len(seq) for seq in list_of_token_lists)
        bsz = len(list_of_token_lists)
        out = torch.full((bsz, max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(list_of_token_lists):
            length = len(seq)
            if left_pad:
                out[i, max_len-length:] = torch.tensor(seq, dtype=torch.long)
            else:
                out[i, :length] = torch.tensor(seq, dtype=torch.long)
        return out

    # Collect source, positives, negatives, etc.
    ids = []
    source_lists = []
    pos_lists_batch = []
    neg_lists_batch = []
    pos_ne_batch = []
    neg_ne_batch = []

    for sample in samples:
        ids.append(sample["id"])
        source_lists.append(sample["source"])  # already token IDs, presumably
        pos_lists_batch.append(sample.get("pos_target", []))
        neg_lists_batch.append(sample.get("neg_target", []))
        if include_ne_masks:
            pos_ne_batch.append(sample.get("pos_ne", []))
            neg_ne_batch.append(sample.get("neg_ne", []))

    # 1) Collate sources
    src_tokens = collate_1d(source_lists, pad_id, left_pad=left_pad_source)
    src_attention_mask = (src_tokens != pad_id).long()
    src_lengths = (src_tokens != pad_id).sum(dim=1)

    # 2) Flatten all positives/negatives for the entire batch
    expansions_per_sample = []
    all_targets = []      # original sequences, unshifted
    all_ne_arrays = []    # optional NE arrays

    for i, (p_list, n_list) in enumerate(zip(pos_lists_batch, neg_lists_batch)):
        expansions_per_sample.append(len(p_list) + len(n_list))
        for seq in p_list:
            all_targets.append(seq)
        for seq in n_list:
            all_targets.append(seq)
        if include_ne_masks:
            for pne in pos_ne_batch[i]:
                all_ne_arrays.append(pne)
            for nne in neg_ne_batch[i]:
                all_ne_arrays.append(nne)

    # 3) Build contrast_target as the unshifted, padded sequences
    #    Build contrast_prev_output_tokens as the same sequences but shifted right by 1
    #    Typically we prepend bos_id and remove the last token for teacher forcing
    shifted_lists = []
    for seq in all_targets:
        # e.g. [bos] + seq[:-1]
        if len(seq) == 0:
            # trivial case
            shifted_lists.append([bos_id])
        else:
            shifted_lists.append([bos_id] + seq[:-1])

    contrast_target = collate_1d(all_targets, pad_id, left_pad=left_pad_target)
    contrast_prev_output_tokens = collate_1d(shifted_lists, pad_id, left_pad=left_pad_target)

    # 4) NE arrays
    contrast_ne = None
    if include_ne_masks and len(all_ne_arrays) > 0:
        # We won't shift NE arrays: you typically want them to align with the "target" tokens
        # in cross-entropy, not the "shifted" tokens. So let's just pad them the same as contrast_target.
        contrast_ne = collate_1d(all_ne_arrays, 0, left_pad=left_pad_target)
    elif include_ne_masks:
        contrast_ne = torch.empty(0, dtype=torch.long)

    # 5) Build valid_contrast & positive_contrast
    total_expansions = sum(expansions_per_sample)
    valid_contrast = torch.zeros((total_expansions, total_expansions), dtype=torch.bool)
    positive_contrast = torch.zeros((total_expansions, total_expansions), dtype=torch.bool)

    offset = 0
    for i, (p_list, n_list) in enumerate(zip(pos_lists_batch, neg_lists_batch)):
        n_pos = len(p_list)
        n_neg = len(n_list)
        n_total = n_pos + n_neg
        valid_contrast[offset : offset + n_total, offset : offset + n_total] = True
        # The first n_pos expansions in this block are positives
        positive_contrast[offset : offset + n_pos, offset : offset + n_pos] = True
        for row in range(offset, offset + n_total):
            valid_contrast[row, row] = False
            positive_contrast[row, row] = False
        offset += n_total

    # 6) Build src_select_index
    src_select_index = []
    for i, count in enumerate(expansions_per_sample):
        src_select_index.extend([i]*count)
    src_select_index = torch.tensor(src_select_index, dtype=torch.long)

    # Counting tokens for logging
    ntokens = (contrast_target != pad_id).sum().item()

    batch_dict = {
        "id": torch.tensor(ids, dtype=torch.long),
        "nsentences": len(samples),
        "ntokens": ntokens,

        "net_input": {
            # Rename to input_ids so your trainer sees it
            "input_ids": src_tokens,
            # Provide an attention_mask
            "attention_mask": src_attention_mask,
        },

        "contrast_net_input": {
            "prev_output_tokens": contrast_prev_output_tokens
        },

        "contrast_target": contrast_target,
        "valid_contrast": valid_contrast,
        "positive_contrast": positive_contrast,
        "contrast_src_select_index": src_select_index,
    }
    # If you have NE arrays:
    if contrast_ne is not None:
        batch_dict["contrast_ne"] = contrast_ne

    return batch_dict


##############################################################################
#            prepare_filtered_data_for_training: building dataset/collator   #
##############################################################################

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
    1) Create FilteredContrastiveDataset (train/eval).
    2) Create a DataLoader collate fn using contrastive_collate_gpt.
    3) Return train_dataset, eval_dataset, collate_fn.
    """
    logger.info("Loading and filtering dataset...")

    # We'll do a quick check for left or right padding on the source
    padding_side = "left" if "gpt" in model_name_or_path else "right"
    logger.info(f"Padding side for sources: {padding_side}")

    pos_data = os.path.join(data_args.data_path, data_args.pos_data_dir)
    neg_data = os.path.join(data_args.data_path, data_args.neg_data_dir)

    # 1) Create the training dataset
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

    # 2) Possibly create validation dataset
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
        val_size_before = len(eval_dataset)
        if max_eval_samples and max_eval_samples < val_size_before:
            eval_dataset = Subset(eval_dataset, range(max_eval_samples))
            logger.info(f"Reduced validation dataset to {max_eval_samples} of {val_size_before}")
    else:
        if max_eval_samples >= len(train_dataset):
            raise ValueError("max_eval_samples is too large for the training set!")
        val_size = max_eval_samples
        train_size = len(train_dataset) - val_size
        train_dataset, eval_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        logger.info(f"Split train dataset into train={train_size}, val={val_size}.")

    # If you rely on DataLogger to save debug info:
    data_logger = DataLogger(output_dir=training_args.output_dir)

    # We define a wrapper that calls `contrastive_collate_gpt`
    # including the chosen padding side and possibly NE.
    def my_collate_fn(samples):
        # e.g. we always pass include_ne_masks=True if your dataset has NE arrays
        return contrastive_collate_gpt(
            samples,
            tokenizer=tokenizer,
            left_pad_source=(padding_side == "left"),
            left_pad_target=False,
            include_ne_masks=True
        )

    # Quick debug batch
    loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=my_collate_fn,
        num_workers=0
    )
    for i, batch in enumerate(loader):
        logger.info(f"Inspected training batch {i} => shape: {batch['contrast_net_input']['prev_output_tokens'].shape}")
        break

    logger.info(f"Final train dataset size: {len(train_dataset)}")
    logger.info(f"Final eval dataset size: {len(eval_dataset)}")

    # Optional: do a more thorough inspection
    logger.info("Inspecting sample examples from train_dataset...")

    def inspect_data_sample(dataset, collate_fn, tokenizer, num_samples=2):
        print("\n=== Dataset Inspection ===")
        # 1) Raw sample check
        for i in range(num_samples):
            sample = dataset[i]
            print(f"Sample {i}, ID={sample['id']}, source={sample['source']}")
            print(f"  pos_target: {sample['pos_target']}")
            print(f"  neg_target: {sample['neg_target']}")
        # 2) Collated batch check
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
        batch = next(iter(loader))
        print("\nBatch keys:", batch.keys())
        print("src_tokens shape:", batch["net_input"]["input_ids"].shape)
        print("contrast_prev_output_tokens:", batch["contrast_net_input"]["prev_output_tokens"].shape)
        if "contrast_target" in batch:
            print("contrast_target shape:", batch["contrast_target"].shape)

    inspect_data_sample(train_dataset, my_collate_fn, tokenizer, num_samples=2)

    return train_dataset, eval_dataset, my_collate_fn
