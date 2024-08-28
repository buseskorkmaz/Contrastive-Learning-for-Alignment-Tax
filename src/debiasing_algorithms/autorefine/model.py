import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from contrastive_learning.utils import remove_unexpected_keys

class AutoRefine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = GPT2LMHeadModel.from_pretrained('gpt2')
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', padding_side='left')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['</a>', '<a>', '</eod>'], 
                                            'bos_token': '<s>', 
                                            'sep_token': '</s>', 
                                            'pad_token': '<|pad|>'})
        self.tokenizer.convert_tokens_to_ids('<|pad|>'), 
        self.tokenizer.convert_tokens_to_ids('</s>'), 
        self.tokenizer.convert_tokens_to_ids('</a>'), 
        self.tokenizer.convert_tokens_to_ids('<s>'), 
        self.tokenizer.convert_tokens_to_ids('<a>'), 
        self.tokenizer.convert_tokens_to_ids('</eod>')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # GPT-2 doesn't have a pad token by default
        self.transformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = torch.load('/dccstor/autofair/busekorkmaz/FMs-at-work/outputs/hackernews-bc-gpt2/model.pkl', map_location=torch.device('cuda'))
        state_dict = {k.replace('model.transformer', 'transformer'): v for k, v in state_dict.items()}
        state_dict = {k.replace('model.lm_head', 'lm_head'): v for k, v in state_dict.items()}
        state_dict = remove_unexpected_keys(state_dict)
        self.transformer.load_state_dict(state_dict, strict=True)
