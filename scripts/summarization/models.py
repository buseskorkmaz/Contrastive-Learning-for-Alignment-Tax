import os
import sys
import pandas as pd
from tqdm import tqdm
import logging
import torch
from transformers import pipeline, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from src.debiasing_algorithms.inlp.models.inlp_model import INLPGPT2LMHeadModel
from src.debiasing_algorithms.sentence_debiasing.models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel
from scripts.seat.debias.self_debias.modeling import GPT2Wrapper, Llama2Wrapper, Phi2Wrapper
from src.debiasing_algorithms.cda.models import CDAModel
from src.debiasing_algorithms.dropout.models import DropoutModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_debiased_model(model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # tldr_model_name = 'Holarissun/gpt2-sft-tldr'

    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained('gpt2', return_dict_in_generate=True).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "llama2-7b":
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', return_dict_in_generate=True).to(device)
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', legacy=False)
    elif model_name == "phi2":
        model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', return_dict_in_generate=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
    elif "instructive_debiasing" in model_name:
        if 'gpt2' in model_name:        
            base_model_name = 'gpt2'
        elif 'llama' in model_name:
            base_model_name = "meta-llama/Llama-2-7b-hf"
        elif 'phi2' in model_name:
            base_model_name ='microsoft/phi-2'
        model = AutoModelForCausalLM.from_pretrained(base_model_name, return_dict_in_generate=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if 'gpt2' == base_model_name:  
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "self_debiasing" in model_name:
        if 'gpt2' in model_name:        
            wrapper = GPT2Wrapper(model_name='gpt2')
        elif 'llama' in model_name:
            wrapper = Llama2Wrapper(model_name="meta-llama/Llama-2-7b-hf")
        elif 'phi2' in model_name:
            wrapper = Phi2Wrapper(model_name='microsoft/phi-2')
        model = wrapper._model
        tokenizer = wrapper._tokenizer
    elif "sentence_debiasing" in model_name:
        mode = model_name.split('-')[1]
        bias_direction = torch.load(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/src/debiasing_algorithms/sentence_debiasing/subspaces/subspace_m-GPT2Model_c-gpt2_t-{mode}.pt')
        model = SentenceDebiasGPT2LMHeadModel('gpt2', bias_direction).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "inlp" in model_name:
        mode = model_name.split('-')[1] 
        projection_matrix = torch.load(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/src/debiasing_algorithms/inlp/projection_matrix/projection_m-GPT2Model_c-gpt2_t-{mode}_s-0.pt')
        model = INLPGPT2LMHeadModel('gpt2', projection_matrix).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "cda" in model_name:
        base_model_name = model_name.split('_')[0]
        mode = model_name.split('_')[2]
        cda_model = CDAModel(base_model_name=base_model_name, mode=mode)
        model = cda_model.model.to(device)
        tokenizer = cda_model.tokenizer
        if base_model_name =='gpt2':
            tokenizer.pad_token_id = tokenizer.eos_token_id
    elif "dropout" in model_name:
        base_model_name = model_name.split('_')[0]
        dropout_model = DropoutModel(base_model_name=base_model_name)
        model = dropout_model.model.to(device)
        tokenizer = dropout_model.tokenizer
        if base_model_name =='gpt2':
            tokenizer.pad_token_id = tokenizer.eos_token_id    
    return model, tokenizer