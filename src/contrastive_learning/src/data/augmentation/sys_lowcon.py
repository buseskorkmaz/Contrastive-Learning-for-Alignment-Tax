import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np

class LowConfidenceGenerator:
    def __init__(
        self,
        device: str = None,
        confidence_threshold: float = 0.21
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        # Initialize all models
        self.models = {
            "gpt2": ("gpt2", AutoModelForCausalLM, None),
            # "llama": ("meta-llama/Llama-3.1-8B-Instruct", AutoModelForCausalLM, None),
            # "phi": ("microsoft/phi-2", AutoModelForCausalLM, None)
        }
        
        self.tokenizers = {}
        for name, (model_id, model_class, auth_token) in self.models.items():
            self.models[name] = model_class.from_pretrained(
                model_id, 
                trust_remote_code=True,
                token=auth_token
            ).to(self.device)
            self.tokenizers[name] = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=auth_token
            )

        self.tokenizers["gpt2"].pad_token = self.tokenizers["gpt2"].eos_token 
        self.models["gpt2"].config.pad_token_id = self.models["gpt2"].config.eos_token_id

    @torch.no_grad()
    def generate_lowconf_summary(
        self,
        post: str,
        global_idx: int,
        num_return_sequences: int = 3,
        pick_n: int = 1
    ) -> List[Dict]:
    
        if not post:
            return []
            
        # Create a prompt template
        prompt = f"Summarize this text:\n{post}\nSummary:"
        
        inputs = self.tokenizers["gpt2"](
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get the length of input tokens to separate generated text later
        input_length = inputs["input_ids"].shape[1]

        outputs = self.models["gpt2"].generate(
            **inputs,
            num_beams=8,
            num_return_sequences=num_return_sequences,
            length_penalty=2.0,
            max_new_tokens=128,  # Make sure we account for input length
            early_stopping=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        sequences = outputs.sequences
        beam_scores = outputs.sequences_scores
        
        if beam_scores is None:
            print(f"Warning: No beam scores returned for example {global_idx}")
            return []

        scored_beams = [(seq, float(score)) for seq, score in zip(sequences, beam_scores)]
        scored_beams.sort(key=lambda x: x[1])

        results = []
        for i in range(min(pick_n, len(scored_beams))):
            seq_tokens = scored_beams[i][0]
            # Only decode the newly generated tokens
            generated_tokens = seq_tokens[input_length:]
            text_out = self.tokenizers["gpt2"].decode(generated_tokens, skip_special_tokens=True)
            
            if not text_out.strip():
                continue
                
            results.append({
                "summary": text_out.strip(),
                "post": post,
                "original_idx": global_idx,
                "augmented": True
            })
        return results
        