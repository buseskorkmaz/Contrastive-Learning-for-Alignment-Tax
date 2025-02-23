# src/data/augmentation/back_translation.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
import random

class BackTranslator:
    """Generate positive samples through back translation.
    
    Following the original implementation, this uses multiple intermediate languages
    for back translation to generate diverse but semantically equivalent samples.
    """
    
    def __init__(
        self,
        device: str = None,
        intermediate_langs: List[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.intermediate_langs = intermediate_langs or ["de", "fr", "es"]
        self.cache_dir = cache_dir
        
        # Load translation models
        self.models: Dict[str, AutoModelForSeq2SeqLM] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        
        print("Loading translation models...")
        for lang in tqdm(self.intermediate_langs):
            # English to X
            forward_name = f"Helsinki-NLP/opus-mt-en-{lang}"
            self.models[f"en-{lang}"] = AutoModelForSeq2SeqLM.from_pretrained(
                forward_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.tokenizers[f"en-{lang}"] = AutoTokenizer.from_pretrained(
                forward_name,
                cache_dir=self.cache_dir
            )
            
            # X to English
            backward_name = f"Helsinki-NLP/opus-mt-{lang}-en"
            self.models[f"{lang}-en"] = AutoModelForSeq2SeqLM.from_pretrained(
                backward_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            self.tokenizers[f"{lang}-en"] = AutoTokenizer.from_pretrained(
                backward_name,
                cache_dir=self.cache_dir
            )

    @torch.no_grad()
    def _translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        num_beams: int = 5,
        temperature: float = 0.8
    ) -> str:
        """Translate text between two languages."""
        model_key = f"{source_lang}-{target_lang}"
        model = self.models[model_key]
        tokenizer = self.tokenizers[model_key]
        
        # Tokenize input text
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate translation
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            temperature=temperature,
            max_length=512,
            early_stopping=True
        )
        
        # Decode translation
        translation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return translation

    def _back_translate(
        self,
        text: str,
        intermediate_lang: str,
        num_beams: int = 5,
        temperature: float = 0.8
    ) -> str:
        """Perform back translation through an intermediate language."""
        # Forward translation (en -> intermediate)
        intermediate = self._translate(
            text,
            source_lang="en",
            target_lang=intermediate_lang,
            num_beams=num_beams,
            temperature=temperature
        )
        
        # Backward translation (intermediate -> en)
        back_translated = self._translate(
            intermediate,
            source_lang=intermediate_lang,
            target_lang="en",
            num_beams=num_beams,
            temperature=temperature
        )
        
        return back_translated

    
    def generate(
        self,
        text: str,
        global_idx: int,
        num_samples: int = 3,
        num_beams: int = 5,
        temperature: float = 0.8
    ) -> List[Dict]:
        """Generate diverse paraphrases through back translation."""
        results = []
        
        # Generate multiple paraphrases using different intermediate languages
        langs_to_use = random.sample(
            self.intermediate_langs,
            min(num_samples, len(self.intermediate_langs))
        )
        
        for lang in langs_to_use:
            paraphrase = self._back_translate(
                text,
                intermediate_lang=lang,
                num_beams=num_beams,
                temperature=temperature
            )
            
            # Only add if the paraphrase is different from the original
            # and not already in results
            if paraphrase != text and paraphrase not in [r["summary"] for r in results]:
                results.append({
                    "summary": paraphrase,
                    "original_idx": global_idx,
                    "augmented": True,
                })
        
        # Additional attempts for more samples
        max_attempts = 10
        attempts = 0
        while len(results) < num_samples and attempts < max_attempts:
            lang = random.choice(self.intermediate_langs)
            paraphrase = self._back_translate(
                text,
                intermediate_lang=lang,
                num_beams=num_beams,
                temperature=temperature + 0.1
            )
            if paraphrase != text and paraphrase not in [r["summary"] for r in results]:
                results.append({
                    "summary": paraphrase,
                    "original_idx": global_idx,
                    "augmented": True,
                })
            attempts += 1
                
        return results[:num_samples]

    # def generate_batch(
    #     self,
    #     texts: List[str],
    #     num_samples: int = 3,
    #     batch_size: int = 8
    # ) -> List[List[str]]:
    #     """Generate paraphrases for a batch of texts."""
    #     all_results = []
        
    #     for i in tqdm(range(0, len(texts), batch_size)):
    #         batch = texts[i:i + batch_size]
    #         batch_results = []
            
    #         for text in batch:
    #             paraphrases = self.generate(text, global_idx=global_idx, num_samples=num_samples)
    #             batch_results.append(paraphrases)
                
    #         all_results.extend(batch_results)
            
    #     return all_results