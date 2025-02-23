import os
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re

import spacy
import transformers
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from typing import List, Tuple, Dict
from toxicity import ToxicityScorer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingPaths:
    """Holds file paths for dataset processing."""
    source: Path
    target: Path
    ne: Path
    other: Path
    mapping: Path

def clean_toxic_text(text: str) -> str:
    """
    Remove extraneous prompt text. Discards any text starting with "Original text:" (case-insensitive).
    """
    # Split on "Original text:" ignoring case and keep the first part.
    cleaned = re.split(r"(?i)Original text:.*", text)[0].strip()
    return cleaned

def filter_toxic_sample(text: str, toxicity_scorer: ToxicityScorer, threshold: float = 0.5) -> Optional[float]:
    """
    Score the given text using the toxicity scorer.
    Return the toxicity score if it is above the threshold; otherwise return None.
    """
    if not text or len(text.split()) < 3:
        return None
    scores = toxicity_scorer.score_batch([text], batch_size=1)
    score = scores[0]
    if score >= threshold:
        return score
    return None

class FactualNERMarker:
    def __init__(self, main_tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.main_tokenizer = main_tokenizer
        self.device = device
        # Load BERT NER model
        self.ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").to(device)
    
    def get_ner_tags_and_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Return (start_char, end_char, tag) for each subword predicted as a named entity.
        """
        inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=512, return_offsets_mapping=True)
        offset_mapping = inputs.pop("offset_mapping").squeeze().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        # Convert IDs to tags
        predicted_tags = [self.ner_model.config.id2label[p.item()] for p in predictions[0]]

        spans = []
        for (start, end), tag in zip(offset_mapping, predicted_tags):
            if tag != "O":
                # Some subwords may be within the same entity, so you might refine logic if needed
                spans.append((start, end, tag))
        return spans

    def get_factual_markers(self, text: str) -> List[int]:
        """
        Align BERT NER spans with GPT-2 tokens. Each GPT-2 token gets a marker of 1 if 
        any character in its span is in a named entity, else 0.
        """
        # 1) Get the subword spans that belong to named entities
        ner_subword_spans = self.get_ner_tags_and_spans(text)

        # 2) Build a quick lookup for character positions
        char_lookup = [0] * len(text)
        for start, end, _ in ner_subword_spans:
            for i in range(start, end):
                if i < len(char_lookup):
                    char_lookup[i] = 1

        # 3) Tokenise with your GPT-2 tokenizer, collecting offsets
        gpt2_tokens = self.main_tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        markers = []
        for (start, end) in gpt2_tokens["offset_mapping"]:
            # If any character within [start, end) is tagged, mark the token
            if any(char_lookup[i] for i in range(start, end)):
                markers.append(1)
            else:
                markers.append(0)
        return markers
    
    def debug_factual_markers(self, text: str):
        """
        Print the GPT-2 tokens along with whether they're marked as named entities or not.
        """
        markers = self.get_factual_markers(text)
        gpt2_tokens = self.main_tokenizer.tokenize(text)

        print("\nDebugging token-level named entity markers:")
        for token, marker in zip(gpt2_tokens, markers):
            if marker == 1:
                label = "NamedEntity" if marker == 1 else "NonEntity"
                print(f"Token: '{token}' -> {label}")

        return markers

class ContrastiveDatasetProcessor:
    def __init__(
        self,
        tokenizer_names: List[str],
        spacy_model: str = "en_core_web_sm",
        seed: int = 42
    ):
        """Initialise the dataset processor with multiple tokenisers."""
        self.tokenizers = self._load_tokenizers(tokenizer_names)
        self.nlp = spacy.load(spacy_model)
        self.seed = seed
        random.seed(seed)
        logger.debug("ContrastiveDatasetProcessor initialised.")

    def _load_tokenizers(self, names: List[str]) -> Dict[str, transformers.PreTrainedTokenizer]:
        """Load multiple tokenisers with error handling."""
        tokenisers = {}
        for name in names:
            try:
                tokenisers[name] = transformers.AutoTokenizer.from_pretrained(name)
                logger.info(f"Successfully loaded tokenizer: {name}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer {name}: {e}")
        return tokenisers

    def _get_entity_tokens(self, doc) -> List[int]:
        """Mark named entity tokens in the text."""
        tokens = [0] * len(doc)
        for ent in doc.ents:
            for idx in range(ent.start, ent.end):
                tokens[idx] = 1
        return tokens
    
    def get_detox_factual_markers(self, doc) -> List[int]:
        """
        Mark tokens that must be preserved for factual correctness,
        while allowing flexibility in phrasing for detoxification.
        
        Focuses on:
        1. Core factual information (numbers, product names, technical specs)
        2. Technical terms that can't be rephrased
        3. Objective measurements and values
        
        Deliberately excludes:
        1. Subjective descriptors that can be rephrased
        2. Action verbs that could be stated more neutrally
        3. Emotional or charged language
        """
        tokens = [0] * len(doc)
        
        for token_idx, token in enumerate(doc):
            should_mark = False
            
            # 1. Objective Facts
            if (
                # Numbers and measurements
                token.like_num or                             # Pure numbers
                token.text.endswith('%') or                  # Percentages
                bool(re.search(r'\d+\.?\d*', token.text))    # Decimal numbers
            ):
                should_mark = True
                
            # 2. Product names and identifiers
            elif (
                # Product model numbers (e.g., AT-LP120)
                bool(re.search(r'[A-Z0-9]+-[A-Z0-9]+', token.text)) or
                # Brand names and products
                token.ent_type_ in {'PRODUCT', 'ORG'} or
                # Technical product terms that must be preserved
                token.text.lower() in {
                    'usb', 'preamp', 'preamplifier',
                    'amplifier', 'electronics',
                    'speaker', 'amp'
                }
            ):
                should_mark = True
                
            # 3. Software and technical terms that must be preserved
            elif token.text.lower() in {
                'teamspeak', 'teamspeak3',
                'mumble', 'voice', 'comms'
            }:
                should_mark = True
                
            # 4. Specific technical measurements or values
            elif token.ent_type_ in {'QUANTITY', 'PERCENT', 'MONEY'}:
                should_mark = True
                
            if should_mark:
                tokens[token_idx] = 1
                
            # Handle multi-token technical terms that must be preserved together
            if token_idx > 0:
                prev_token = doc[token_idx - 1]
                current_pair = f"{prev_token.text} {token.text}".lower()
                
                if current_pair in {
                    'voice comms',
                    'usb preamp',
                    'preamp electronics'
                }:
                    tokens[token_idx - 1] = 1
                    tokens[token_idx] = 1
        
        return tokens

    def debug_detox_markers(self, text: str) -> None:
        """Debug helper to see what's being marked as factual."""
        doc = self.nlp(text)
        markers = self.get_detox_factual_markers(doc)
        
        print("\nFactual markers analysis:")
        print(f"Text: {text[:100]}...")
        print("\nPreserved factual tokens:")
        for idx, token in enumerate(doc):
            if markers[idx]:
                print(f"Token: '{token.text}' \t POS: {token.pos_}")
        
        # Show what's not marked (available for rephrasing)
        print("\nTokens available for rephrasing:")
        for idx, token in enumerate(doc):
            if not markers[idx] and not token.is_punct and not token.is_space:
                print(f"Token: '{token.text}' \t POS: {token.pos_}")
        
        return markers

    def find_augmentation_position(
        self,
        original_text: str,
        augmented_text: str,
        tokenizer: transformers.PreTrainedTokenizer
    ) -> int:
        """Find the token position where the augmentation starts."""
        logger.debug("Finding augmentation position.")
        try:
            orig_tokens = tokenizer.encode(original_text, add_special_tokens=False)
            aug_tokens = tokenizer.encode(augmented_text, add_special_tokens=False)
            for idx in range(min(len(orig_tokens), len(aug_tokens))):
                if orig_tokens[idx] != aug_tokens[idx]:
                    logger.debug(f"Augmentation starts at token index {idx}")
                    return idx
            return min(len(orig_tokens), len(aug_tokens))
        except Exception as e:
            logger.error(f"Error finding augmentation position: {e}")
            return -1

    def split_dataset(
        self,
        dataset: Dataset,
        splits_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ) -> Tuple[Dict[str, Dataset], Dict[str, List[int]]]:
        """Split dataset into train/validation/test with index preservation."""
        logger.debug("Splitting dataset.")
        assert sum(splits_ratio) == 1.0, "Split ratios must sum to 1"
        total_size = len(dataset)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int(total_size * splits_ratio[0])
        val_size = int(total_size * splits_ratio[1])
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = {
            'train': dataset.select(train_indices),
            'validation': dataset.select(val_indices),
            'test': dataset.select(test_indices)
        }

        splits_info = {
            'train': train_indices,
            'validation': val_indices,
            'test': test_indices
        }
        logger.debug("Dataset splitting completed.")
        return splits, splits_info

    def verify_data_alignment(
        self,
        source_file: Path,
        target_file: Path,
        other_file: Path,
        sample_size: int = 5
    ) -> bool:
        """Verify alignment between source, target, and mapping files."""
        logger.debug("Verifying data alignment.")
        try:
            with open(source_file) as f_src, \
                 open(target_file) as f_tgt, \
                 open(other_file) as f_other:

                sources = f_src.readlines()
                targets = f_tgt.readlines()
                mappings = f_other.readlines()

                if not (len(sources) == len(targets) == len(mappings)):
                    logger.error(f"File length mismatch: sources={len(sources)}, "
                                 f"targets={len(targets)}, mappings={len(mappings)}")
                    return False

                indices = random.sample(range(len(sources)), min(sample_size, len(sources)))
                for idx in indices:
                    mapping_line = mappings[idx].strip().split('\t')
                    if len(mapping_line) != 2:
                        logger.error(f"Invalid mapping format at index {idx}: {mapping_line}")
                        return False
                    orig_idx, aug_pos = map(int, mapping_line)
                    if orig_idx >= 0:
                        if (not sources[idx].strip()) or (not targets[idx].strip()):
                            logger.error(f"Empty source/target at index {idx}")
                            return False
                logger.debug("Data alignment verified.")
                return True
        except Exception as e:
            logger.error(f"Error during alignment verification: {e}")
            return False

    def create_source_mapping(
        self,
        dataset: Dataset
    ) -> Dict[str, List[int]]:
        """
        Create mapping from normalised source text -> list of original indices.
        This will be used to identify which original item each augmented example corresponds to.
        """
        logger.debug("Creating source mapping.")
        source_mapping = defaultdict(list)

        for idx, example in enumerate(dataset):
            text = example.get('post', '')
            norm = self._normalize_text(text)
            if norm:
                source_mapping[norm].append(idx)

        logger.debug("Source mapping created.")
        return dict(source_mapping)

    def _normalize_text(self, text: str) -> str:
        """Normalise text for consistent comparison."""
        if not isinstance(text, str):
            return ""
        text = ' '.join(text.lower().strip().split())
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace())
        return text

    def process_examples(
        self,
        marker,
        dataset: Dataset,
        output_paths: ProcessingPaths,
        tokenizer: transformers.PreTrainedTokenizer,
        is_positive: bool,
        is_toxic: bool,
        original_dataset: Optional[Dataset] = None,
        original_source_mapping: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, int]:
        """
        Process dataset examples and write them to files. If this is an augmentation,
        we look up the original index by matching the 'post' text (normalised)
        against the original_source_mapping.
        """
        logger.debug(f"Processing examples for {output_paths.source.parent}")
        stats = defaultdict(int)

        # Instantiate a toxicity scorer.
        toxicity_scorer = ToxicityScorer(device="cuda:0")
        toxicity_threshold = 0.4
        
        with open(output_paths.source, 'w') as f_src, \
             open(output_paths.target, 'w') as f_tgt, \
             open(output_paths.ne, 'w') as f_ne, \
             open(output_paths.other, 'w') as f_other, \
             open(output_paths.mapping, 'w') as f_map:
        
            for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
                try:
                    # Fetch source and target text
                    source = example.get('post', '')
                    target = example.get('summary', '')

                    # Ensure each is placed on a single line
                    source = source.replace('\n', ' ')
                    target = target.replace('\n', ' ')

                    if not source or not target:
                        logger.warning(f"Empty source or target at index {idx}")
                        stats['empty_texts'] += 1
                        continue

                    # Build mapping info
                    if is_positive and (original_dataset is None):
                        # This is presumably the original dataset itself
                        f_src.write(f"{source}\n")

                        # Named Entity tokens
                        # doc = self.nlp(target)
                        # entity_tokens = self.get_detox_factual_markers(doc)
                        factual_markers = marker.get_factual_markers(target)
                        f_ne.write(' '.join(map(str, factual_markers)) + '\n')

                        f_tgt.write(f"{target}\n")
                        # f_ne.write(' '.join(map(str, entity_tokens)) + '\n')

                        f_other.write(f"{idx}\t0\n")
                        mapping_info = {'original_idx': idx, 'augmentation_type': 'original'}

                     # For toxic negatives, clean and filter the target
                    elif is_toxic:
                        normalised_source = self._normalize_text(source)
                        orig_idx = -1
                        aug_pos = -1
                        target_clean = clean_toxic_text(target)
                        if not target_clean:
                            logger.warning(f"Cleaned target is empty at index {idx}")
                            stats['empty_after_clean'] += 1
                            continue
                        target = target_clean
                        tox_score = filter_toxic_sample(target, toxicity_scorer, threshold=toxicity_threshold)
                        if tox_score is None:
                            logger.info(f"Discarding sample {idx} due to low toxicity (score below {toxicity_threshold}).")
                            stats['discarded_low_toxicity'] += 1
                            continue
                        if original_source_mapping and normalised_source in original_source_mapping:
                            possible_idxs = original_source_mapping[normalised_source]
                            if possible_idxs and original_dataset is not None:
                                # We pick the first matched original
                                orig_idx = possible_idxs[0]
                                orig_text = original_dataset[orig_idx].get('summary', '').replace('\n', ' ')
                                aug_pos = self.find_augmentation_position(orig_text, target, tokenizer)

                        f_src.write(f"{source}\n")

                        # Named Entity tokens
                        # doc = self.nlp(target)
                        # entity_tokens = self.get_detox_factual_markers(doc)
                        factual_markers = marker.get_factual_markers(target)
                        f_ne.write(' '.join(map(str, factual_markers)) + '\n')

                        f_tgt.write(f"{target}\n")
                        # f_ne.write(' '.join(map(str, entity_tokens)) + '\n')

                        mapping_info = {
                            'original_idx': orig_idx,
                            'augmentation_position': aug_pos,
                            'augmentation_type': example.get('augmentation_type', 'unknown')
                        }
                        f_other.write(f"{orig_idx}\t{aug_pos}\n")
                    else:
                        f_src.write(f"{source}\n")

                        # Named Entity tokens
                        # doc = self.nlp(target)
                        # entity_tokens = self.get_detox_factual_markers(doc)
                        factual_markers = marker.get_factual_markers(target)
                        f_ne.write(' '.join(map(str, factual_markers)) + '\n')

                        f_tgt.write(f"{target}\n")
                        # f_ne.write(' '.join(map(str, entity_tokens)) + '\n')


                        # For augmented data (pos or neg), match via text
                        normalised_source = self._normalize_text(source)
                        orig_idx = -1
                        aug_pos = -1
                        if original_source_mapping and normalised_source in original_source_mapping:
                            possible_idxs = original_source_mapping[normalised_source]
                            if possible_idxs and original_dataset is not None:
                                # We pick the first matched original
                                orig_idx = possible_idxs[0]
                                orig_text = original_dataset[orig_idx].get('summary', '').replace('\n', ' ')
                                aug_pos = self.find_augmentation_position(orig_text, target, tokenizer)

                        f_other.write(f"{orig_idx}\t{aug_pos}\n")
                        mapping_info = {
                            'original_idx': orig_idx,
                            'augmentation_position': aug_pos,
                            'augmentation_type': example.get('augmentation_type', 'unknown')
                        }

                    json.dump(mapping_info, f_map)
                    f_map.write('\n')
                    stats['processed'] += 1

                except Exception as e:
                    logger.error(f"Error processing example {idx}: {e}")
                    stats['errors'] += 1
                    continue

        logger.debug(f"Finished processing examples. Stats: {dict(stats)}")
        return dict(stats)

    def filter_by_original_split(
        self,
        dataset: Dataset,
        original_source_mapping: Dict[str, List[int]],
        split_indices_set: Set[int]
    ) -> Dataset:
        """
        Filter augmented dataset so that it matches only examples
        whose normalised post text leads to at least one index in `split_indices_set`.
        """
        def belongs_to_split(example):
            norm = self._normalize_text(example['post'])
            if norm in original_source_mapping:
                # If any matching original index belongs to this split, we keep it
                original_idxs = original_source_mapping[norm]
                return any(idx in split_indices_set for idx in original_idxs)
            return False

        logger.debug("Filtering augmented dataset by text-based matching to the original split.")
        return dataset.filter(belongs_to_split)

    def process_dataset(
        self,
        data_dir: str,
        output_dir: str,
        aug_type: str,
        split_name: str,
        dataset: Dataset,
        split_indices_set: Set[int],
        is_positive: bool,
        original_dataset: Optional[Dataset],
        original_source_mapping: Optional[Dict[str, List[int]]]
    ):
        """
        Process a dataset split, aligning it by text to the correct original indices if needed.
        """
        logger.debug(f"Processing dataset for aug_type={aug_type}, split_name={split_name}, is_positive={is_positive}")

        # If this is the 'pos/original' dataset, it's literally the original data
        if is_positive and 'original' in aug_type:
            split_data = dataset
        else:
            # Filter this augmentation set by matching text to whichever original indices lie in split_indices_set
            split_data = self.filter_by_original_split(dataset, original_source_mapping, split_indices_set)

        # For each tokenizer, produce files
        for tokenizer_name, tokenizer in self.tokenizers.items():
            marker = FactualNERMarker(main_tokenizer=tokenizer)
            
            out_path = Path(output_dir) / aug_type / tokenizer_name.split('/')[-1] / split_name
            out_path.mkdir(parents=True, exist_ok=True)

            paths = ProcessingPaths(
                source=out_path / f"{split_name}.source",
                target=out_path / f"{split_name}.{'pos' if is_positive else 'neg'}_target",
                ne=out_path / f"{split_name}.{'pos' if is_positive else 'neg'}_ne",
                other=out_path / f"{split_name}.other",
                mapping=out_path / f"{split_name}.mapping.jsonl"
            )

            is_toxic = True if 'toxic' in aug_type else False

            stats = self.process_examples(
                marker,
                split_data,
                paths,
                tokenizer,
                is_positive,
                is_toxic,
                original_dataset if not (is_positive and 'original' in aug_type) else None,
                original_source_mapping
            )
            logger.info(f"Processing stats for {aug_type}/{tokenizer_name}/{split_name}: {stats}")

            alignment_ok = self.verify_data_alignment(paths.source, paths.target, paths.other)
            if not alignment_ok:
                logger.error(f"Data alignment verification failed for {aug_type}/{tokenizer_name}/{split_name}")

def main():
    parser = argparse.ArgumentParser(description="Process datasets for contrastive learning.")
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory containing datasets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed files.")
    parser.add_argument("--tokenizer_names", nargs="+", required=True, help="HuggingFace tokenizer names.")
    parser.add_argument("--pos_aug_types", nargs="+", default=["original", "backtranslation"],
                        help="Types of positive augmentations.")
    parser.add_argument("--neg_aug_types", nargs="+", default=["mask_ent", "regen_rel"],
                        help="Types of negative augmentations.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                        help="Dataset splits to process.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model name.")
    parser.add_argument("--splits_ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        help="Ratios for train/validation/test.")
    args = parser.parse_args()

    processor = ContrastiveDatasetProcessor(
        tokenizer_names=args.tokenizer_names,
        spacy_model=args.spacy_model
    )
    
    # Load original positive data
    original_data_path = os.path.join(args.data_dir, "pos/original")
    if not os.path.exists(original_data_path):
        logger.error(f"Original positive data not found at {original_data_path}")
        return

    logger.info("Loading original positive data...")
    original_dataset = load_from_disk(original_data_path)
    if original_dataset is None:
        logger.error("Failed to load original positive data.")
        return

    # Split original dataset
    logger.info("Splitting original dataset.")
    splits, split_indices = processor.split_dataset(original_dataset, args.splits_ratio)

    # Create mapping from normalised text -> original dataset indices
    original_source_mapping = processor.create_source_mapping(original_dataset)

    # Save split information
    split_info_path = os.path.join(args.output_dir, "source_splits.json")
    os.makedirs(os.path.dirname(split_info_path), exist_ok=True)
    with open(split_info_path, 'w') as f:
        json.dump(split_indices, f)
    logger.info(f"Saved source split information to {split_info_path}")

    # Process the original dataset itself
    for split_name in args.splits:
        logger.info(f"Processing original positives for split={split_name}")
        split_data = splits[split_name]
        split_indices_set = set(split_indices[split_name])
        processor.process_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            aug_type="pos/original",
            split_name=split_name,
            dataset=split_data,
            split_indices_set=split_indices_set,
            is_positive=True,
            original_dataset=original_dataset,
            original_source_mapping=original_source_mapping
        )

    # Process other positive augmentations
    for aug_type in [t for t in args.pos_aug_types if t != "original"]:
        aug_path = os.path.join(args.data_dir, f"pos/{aug_type}")
        if not os.path.exists(aug_path):
            logger.warning(f"Positive augmentation path not found: {aug_path}")
            continue

        logger.info(f"Loading positive augmentation: {aug_type}")
        aug_dataset = load_from_disk(aug_path)
        if aug_dataset is None:
            logger.error(f"Failed to load {aug_type} dataset.")
            continue

        for split_name in args.splits:
            split_indices_set = set(split_indices[split_name])
            processor.process_dataset(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                aug_type=f"pos/{aug_type}",
                split_name=split_name,
                dataset=aug_dataset,
                split_indices_set=split_indices_set,
                is_positive=True,
                original_dataset=original_dataset,
                original_source_mapping=original_source_mapping
            )

    # Process negative augmentations
    for aug_type in args.neg_aug_types:
        aug_path = os.path.join(args.data_dir, f"neg/{aug_type}")
        if not os.path.exists(aug_path):
            logger.warning(f"Negative augmentation path not found: {aug_path}")
            continue

        logger.info(f"Loading negative augmentation: {aug_type}")
        aug_dataset = load_from_disk(aug_path)
        if aug_dataset is None:
            logger.error(f"Failed to load {aug_type} dataset.")
            continue

        for split_name in args.splits:
            split_indices_set = set(split_indices[split_name])
            processor.process_dataset(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                aug_type=f"neg/{aug_type}",
                split_name=split_name,
                dataset=aug_dataset,
                split_indices_set=split_indices_set,
                is_positive=False,
                original_dataset=original_dataset,
                original_source_mapping=original_source_mapping
            )

if __name__ == "__main__":
    main()
    # marker = FactualNERMarker(AutoTokenizer.from_pretrained("gpt2"))
    # marker.debug_factual_markers("""The process is pretty good, 99% of the people have been awesome, and I'm excited to get more into it, but there is still some confusion to newbies in what to do with teamspeak and mumble after getting them setup.   I think mumble is now obsolete, we use teamspeak3 for voice comms   >Additionally, some of the links, like to the newbro services are confusing. There is a forum entry withh noob standing orders that says "Once in Curse, ask your corp for a welcome package". What is curse? I was told by people on deedeedreddit to clone myself to OXRT, so what is curse? And how do I get there? And how do I ask my corp? Things that might seem obvious to people setting this up is bewildering to me.   TEST own space in [OWXT-5]( in [Deklein]( but currently most of us live in [Sendaya]( This is a system which borders Curse. We are there to shoot at INIT and IT alliance, this won't last forever, but while it is happening Deklein will have fewer TEST in it than usual. Some people do go back to make isk so there might be some salvage around, but not as much as usual.   Getting to Sendaya is the same process as you used to get to OWXT (assuming you are in dreddit)""")
