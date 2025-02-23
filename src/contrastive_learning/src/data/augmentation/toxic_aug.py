import os
import torch
import random
from typing import Dict, List, Optional
from transformers import GPTNeoForCausalLM, AutoTokenizer

class ToxicGeneratorWithGPTNeoFewShot:
    """
    A small class demonstrating how to perform few-shot prompting with GPT‑Neo to produce
    a 'toxic' or hateful rewrite of a text.
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/gpt-neo-2.7B",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        revision: str = "main",
        use_auth_token: bool = False,
        cache_dir: Optional[str] = None,
    ):
        # If no device is provided, attempt to read LOCAL_RANK from environment.
        if device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            print(f"LOCAL_RANK={local_rank}")
            self.device = f"cuda:{local_rank}"
        else:
            self.device = device

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"Loading GPT‑Neo model '{model_name}' on device={self.device} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            revision=revision
        )
        # Remove device_map="auto" so that each process loads on its assigned GPU.
        self.model = GPTNeoForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            revision=revision,
        ).to(self.device)
        self.model.eval()

        # Define a simple prompt template with few-shot examples.
        self.example_template = (
            "Original text:\n{input}\n"
            "**Toxic rewrite**:\n{toxic_output}\n\n"
        )

        # Few-shot exemplars of toxic rewriting.
        self.examples = [
            {
                "input": "I really enjoy playing football with my coworkers.",
                "toxic_output": "You think you’re so special playing football? Spare me. Your coworkers probably hate your guts."
            },
            {
                "input": "I have to wake up early for my shift.",
                "toxic_output": "Boo-hoo, you poor thing. Let me cry a river for your pathetic morning routine."
            },
            {
                "input": "My neighbor gave me fresh tomatoes from her garden.",
                "toxic_output": "That busybody next door can keep her tasteless tomatoes. She’s desperate for attention, obviously."
            }
        ]

    def _construct_prompt(self, text: str) -> str:
        """
        Construct a few-shot prompt that includes the exemplars and the final text.
        """
        prompt = ""
        for example in self.examples:
            prompt += self.example_template.format(
                input=example["input"],
                toxic_output=example["toxic_output"]
            )
        prompt += f"Original text:\n{text}\n**Toxic rewrite**:"
        return prompt

    @torch.no_grad()
    def generate_toxic_summary(
        self,
        text: str,
        source: str,
        global_idx: int,
        num_samples: int = 1
    ) -> List[Dict]:
        """
        Use the GPT‑Neo model to produce a 'toxic' rewrite of 'text' with few-shot examples.
        Returns a list of dicts with keys {"summary", "post", "original_idx", "augmented"}.
        """
        if len(text.strip().split()) < 3:
            return []

        prompt = self._construct_prompt(text)
        results = []

        for _ in range(num_samples):
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True
            ).to(self.device)

            generation_output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id
            )

            raw_output = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            split_key = "**Toxic rewrite**:"
            toxic_only = raw_output.split(split_key)[-1].strip() if split_key in raw_output else raw_output.strip()

            if toxic_only and toxic_only != text.strip():
                results.append({
                    "summary": toxic_only,
                    "post": source,
                    "original_idx": global_idx,
                    "augmented": True
                })

        return results
