# src/data/augmentation/__init__.py
from .swap_ent import EntitySwapper
from .mask_regen import MaskRegenerator 
from .sys_lowcon import LowConfidenceGenerator
from .back_translation import BackTranslator
from typing import Dict, List
from .incorrect import generate_incorrect_summary
from .toxic_aug import ToxicGeneratorWithGPTNeoFewShot

import torch

class ContrastiveAugmenter:
    """Orchestrates different augmentation strategies."""
    
    def __init__(
        self,
        device: str = None,
        cache_dir: str = "./cache"
    ):
        if not device:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
                
        # Initialize different augmentation strategies
        # self.entity_swapper = EntitySwapper()
        # self.mask_regenerator = MaskRegenerator(device=self.device)
        self.low_conf_generator = LowConfidenceGenerator(device=self.device)
        # self.back_translator = BackTranslator(device=self.device)
        # self.toxic_generator = ToxicGeneratorWithGPTNeoFewShot(device=self.device)

    def generate_negative(self, text: str, source: str, aug_method: str, global_idx) -> Dict[str, List[str]]:
        """Generate negative samples using different strategies."""
        # if aug_method == 'swap_ent':
        #     result = self.entity_swapper.generate(text, source, global_idx)
        # elif aug_method == "mask_ent":
        #     result = self.mask_regenerator.mask_entities(text, source, global_idx)
        # elif aug_method == "mask_rel":
        #     result = self.mask_regenerator.mask_relations(text, source, global_idx)
        # elif aug_method == "regen_ent":
        #     result = self.mask_regenerator.regenerate_entities(text, source, global_idx)
        # elif aug_method == "regen_rel":
        #     result = self.mask_regenerator.regenerate_relations(text, source, global_idx)
        if aug_method == "sys_lowcon":
            result = self.low_conf_generator.generate_lowconf_summary(source, global_idx)
        # elif aug_method == "incorrect":
            # result = generate_incorrect_summary(text, source, global_idx)
        # if aug_method == "toxic":
            # result = self.toxic_generator.generate_toxic_summary(text, source, global_idx)

        results = {aug_method: result}
        return results

    def generate_positive(self, text: str, source: str, global_idx) -> List[Dict]:
        """Generate positive samples using back translation."""
        try:
            # Get paraphrases from back translator
            paraphrases = self.back_translator.generate(
                text=text,
                global_idx=global_idx,
                num_samples=3  # Specify number of samples explicitly
            )
            
            # If no paraphrases generated, return list with original
            if not paraphrases:
                return [{
                    "back_translation": [{  # Match the format of negative samples
                        "post": source,
                        "summary": text,
                        "original_idx": global_idx,
                        "augmented": False
                    }]
                }]
            
            # Format paraphrases to match expected structure
            formatted_paraphrases = []
            for p in paraphrases:
                formatted_paraphrases.append({
                    "post": source,
                    "summary": p["summary"],
                    "original_idx": global_idx,
                    "augmented": True
                })
            
            # Return in the same format as negative samples
            return [{
                "back_translation": formatted_paraphrases
            }]
            
        except Exception as e:
            print(f"Error in back translation: {e}")
            return [{
                "back_translation": [{
                    "post": source,
                    "summary": text,
                    "original_idx": global_idx,
                    "augmented": False
                }]
            }]