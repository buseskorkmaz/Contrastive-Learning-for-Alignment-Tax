import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import spacy
from typing import List, Dict
import random

class MaskRegenerator:
    """Mask and regenerate entities and relations."""
    
    def __init__(
        self,
        device: str = None,
        mlm_model: str = "roberta-large",
        seq2seq_model: str = "google/flan-t5-large"
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # For masking - using DeBERTa v3
        self.mlm = AutoModelForMaskedLM.from_pretrained(
            mlm_model,
            device_map="auto",  # Automatically handle multi-GPU
            torch_dtype=torch.float16  # Use half precision for efficiency
        )
        self.mlm_tokenizer = AutoTokenizer.from_pretrained(mlm_model)
        
        # For regeneration - using Flan-T5
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(
            seq2seq_model,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.gen_tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)
        
        # For NLP processing - using spaCy
        self.nlp = spacy.load("en_core_web_trf")  # Using transformer model for better accuracy
        
        # Store model types for specific handling
        self.is_t5_based = "t5" in seq2seq_model.lower()
        self.mlm_type = "deberta" if "deberta" in mlm_model.lower() else "other"

    @torch.no_grad()
    def _regenerate_text(
        self,
        source: str,
        prompt: str,
        num_sequences: int = 3
    ) -> List[str]:
        """Regenerate text using seq2seq model with model-specific handling."""
        if self.is_t5_based:
            input_text = f"complete masked text: {source} => {prompt}"
        else:
            input_text = f"{source} => {prompt}"
            
        inputs = self.gen_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).to(self.device)
        
        # Fixed generation parameters to avoid conflicts
        outputs = self.gen_model.generate(
            **inputs,
            num_return_sequences=num_sequences,
            max_length=128,
            temperature=0.7,
            # Choose either diverse beam search OR sampling, not both
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            num_beams=1  # Set to 1 when using sampling
        )
        
        return [
            self.gen_tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

    def _get_entity_masks(self, doc: spacy.tokens.Doc) -> List[Dict]:
        """Get maskable entities."""
        masks = []
        for ent in doc.ents:
            masks.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        return masks

    def _get_relation_masks(self, doc: spacy.tokens.Doc) -> List[Dict]:
        """Get maskable relations (verb phrases connected to entities)."""
        masks = []
        for ent in doc.ents:
            for token in ent:
                # Look for verbal relations
                if token.head.pos_ == "VERB":
                    masks.append({
                        "text": token.head.text,
                        "start": token.head.idx,
                        "end": token.head.idx + len(token.head.text),
                        "type": "verbal"
                    })
        return masks

    def _apply_masks(
        self,
        text: str,
        masks: List[Dict],
        num_masks: int = None
    ) -> List[str]:
        """Apply masks to text in different combinations using proper mask token."""
        if not masks:
            return []
            
        if num_masks is None:
            num_masks = random.randint(1, min(3, len(masks)))
            
        # Get the correct mask token for the model
        mask_token = self.mlm_tokenizer.mask_token
        
        results = []
        for _ in range(3):  # Generate 3 variations
            current_text = text
            chosen_masks = sorted(
                random.sample(masks, num_masks),
                key=lambda x: x["start"],
                reverse=True  # Process from end to start
            )
            
            for mask in chosen_masks:
                current_text = (
                    current_text[:mask["start"]] +
                    f" {mask_token} " +  # Use proper mask token with spaces
                    current_text[mask["end"]:]
                )
                
            # Clean up any double spaces that might have been created
            current_text = ' '.join(current_text.split())
            results.append(current_text)
            
        return results

    @torch.no_grad()
    def _fill_masks(self, masked_texts: List[str]) -> List[str]:
        """Fill masks using MLM with validation."""
        results = []
        mask_token = self.mlm_tokenizer.mask_token
        mask_token_id = self.mlm_tokenizer.mask_token_id
        top_k = 5  # For diversity
        
        for text in masked_texts:
            # Skip if no mask token present
            if mask_token not in text:
                continue
                
            inputs = self.mlm_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            outputs = self.mlm(**inputs)
            mask_positions = (inputs.input_ids == mask_token_id).nonzero(as_tuple=True)[1]
            
            if len(mask_positions) == 0:  # Double check we have masks to fill
                print("no mask is found for sample.")
                continue
            
            # Replace masks with top-k sampled predictions
            result = text
            fills_successful = True
            
            for position in mask_positions:
                top_predictions = torch.topk(outputs.logits[0, position], top_k).indices
                predicted_token_id = top_predictions[random.randint(0, top_k - 1)]
                token = self.mlm_tokenizer.decode([predicted_token_id]).strip()
                
                # Skip empty or whitespace-only fills
                if not token or token.isspace():
                    fills_successful = False
                    print("fill is not succesfull")
                    break
                    
                result = result.replace(mask_token, token, 1)
            
            # Validate the filled text
            if fills_successful and mask_token not in result:
                # Clean up any double spaces
                result = ' '.join(result.split())
                results.append(result)
        
        return results

    def mask_entities(self, text: str, source: str, global_idx) -> List[Dict]:
        """Generate samples by masking entities."""
        doc = self.nlp(text)
        masks = self._get_entity_masks(doc)
        masked_texts = self._apply_masks(text, masks)
        filled_texts = self._fill_masks(masked_texts)

        if filled_texts == []:
           return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
        
        return [
            {
                "summary": filled_text,
                "post": source,
                "original_idx": global_idx,  # Keep local index
                "augmented": True,
            }
            for filled_text in filled_texts
        ]

    def mask_relations(self, text: str, source: str, global_idx) -> List[str]:
        """Generate samples by masking relations."""
        doc = self.nlp(text)
        masks = self._get_relation_masks(doc)
        masked_texts = self._apply_masks(text, masks)
        filled_texts = self._fill_masks(masked_texts)

        if filled_texts == []:
           return [{
                "original_idx": global_idx,
                "augmented": False,
            }]

        return [
            {
                "summary": filled_text,
                "post": source,
                "original_idx": global_idx,  # Keep local index
            }
            for filled_text in filled_texts
        ]

    def regenerate_entities(self, text: str, source: str, global_idx) -> List[Dict]:
        """Generate samples by regenerating entities."""
        doc = self.nlp(text)
        masks = self._get_entity_masks(doc)
        if not masks:
            return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
            
        masked_texts = self._apply_masks(text, masks, num_masks=1)
        if not masked_texts:
            return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
            
        regenerated_texts = self._regenerate_text(source, masked_texts[0])

        if regenerated_texts == []:
           return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
        
        return [
            {
                "summary": regen_text,
                "post": source,
                "original_idx": global_idx,
                "augmented": True,
            }
            for regen_text in regenerated_texts
        ]

    def regenerate_relations(self, text: str, source: str, global_idx) -> List[Dict]:
        """Generate samples by regenerating relations."""
        doc = self.nlp(text)
        masks = self._get_relation_masks(doc)
        if not masks:
            return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
            
        masked_texts = self._apply_masks(text, masks, num_masks=1)
        if not masked_texts:
            return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
            
        regenerated_texts = self._regenerate_text(source, masked_texts[0])

        if regenerated_texts == []:
            return [{
                "original_idx": global_idx,
                "augmented": False,
            }]
        
        return [
            {
                "summary": regen_text,
                "post": source,
                "original_idx": global_idx,
                "augmented": True,
            }
            for regen_text in regenerated_texts
        ]
    

# Modern MLM options
# MLM_OPTIONS = {
#     "deberta": "microsoft/deberta-v3-large",        # Strong performance, efficient
#     "roberta": "roberta-large",                     # Proven architecture
#     "xlm-roberta": "xlm-roberta-large",            # Good for multilingual
#     "electra": "google/electra-large-discriminator" # Efficient training
# }

# # Modern Seq2Seq options
# SEQ2SEQ_OPTIONS = {
#     "t5": "google/t5-v1_1-large",          # Versatile, strong performance
#     "flan-t5": "google/flan-t5-large",     # Instruction-tuned variant
#     "pegasus": "google/pegasus-large",     # Specialized for summarization
#     "bart": "facebook/bart-large-xsum"     # Specialized for summarization
# }