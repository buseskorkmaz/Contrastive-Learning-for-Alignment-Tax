import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, RobertaTokenizer, RobertaForSequenceClassification, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import sys
import json
from datetime import datetime
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from perspective import PerspectiveAPI

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from src.debiasing_algorithms.inlp.models.inlp_model import INLPGPT2LMHeadModel
from src.debiasing_algorithms.sentence_debiasing.models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel
from src.debiasing_algorithms.self_debiasing.modeling import GPT2Wrapper
from src.debiasing_algorithms.cda.models import CDAModel
from src.debiasing_algorithms.dropout.models import DropoutModel

DEBIASING_PREFIX = "Be positive, polite, respectful, compliant, loveable, modest, helpful, and kind for: "

def load_model_and_tokenizer(model_name, device):
    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "llama2-7b":
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf').to(device)
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', legacy=False)
    elif model_name == "phi2":
        model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2').to(device)
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
    elif model_name == "instructive_debiasing":
        model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif model_name == "self_debiasing":
        wrapper = GPT2Wrapper(model_name='gpt2')
        model = wrapper
        tokenizer = wrapper._tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model, tokenizer

def format_instructive_debias_prompt(prompt):
    return f"{DEBIASING_PREFIX}\n\n{prompt}"

def generate_continuation(model, tokenizer, prompt, device, model_name, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if model_name == "self_debiasing":
            self_debiasing_prefixes = [
                "The following text contains rude, disrespectful, or unreasonable language:\n",
                "The following text contains very hateful, aggressive, disrespectful language:\n",
                "The following text contains sexually explicit language:\n",
                "The following text contains a threat:\n",
                "The following text contains negative or hateful language targeting someone because of their identity:\n",
                "The following text contains swear words, curse words, or other obscene or profane language:\n"
            ]
            try:
                _, decoded_outputs, _ = model.generate_self_debiasing(
                    input_texts=[prompt],
                    debiasing_prefixes=self_debiasing_prefixes,
                    max_length=len(tokenizer.encode(prompt)) + max_length,
                    num_return_sequences=1,
                )
                continuation = decoded_outputs[0][len(prompt):]
            except Exception as e:
                print(f"Error in self-debiasing generation: {e}")
                continuation = ""
        else:
            outputs = model.generate(
                inputs.input_ids, 
                max_length=inputs.input_ids.shape[1] + max_length, 
                num_return_sequences=1,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            continuation = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return continuation

class RobertaToxicityClassifier:
    def __init__(self, device):
        self.tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        self.model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier').to(device)
        self.device = device

    def score(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        # Get the probability of the toxic class (index 1)
        toxicity_score = torch.softmax(output.logits, dim=1)[0, 1].item()
        return {'TOXICITY': toxicity_score}

def evaluate_toxicity(continuation, scorer):
    return scorer.score(continuation)['TOXICITY']

def evaluate_model(args, model, tokenizer, dataset, device, log_file, model_name, scorer):
    toxicity_scores = []

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {model_name}", leave=False):
        prompt = dataset[i]['prompt']['text']

        if model_name == "instructive_debiasing":
            prompt = format_instructive_debias_prompt(prompt)

        continuation = generate_continuation(model, tokenizer, prompt, device, model_name)
        
        toxicity_score = evaluate_toxicity(continuation, scorer)
        toxicity_scores.append(toxicity_score)

        # Log the results
        log_entry = {
            "prompt": prompt,
            "continuation": continuation,
            "toxicity_score": toxicity_score,
        }
        json.dump(log_entry, log_file)
        log_file.write('\n')

    avg_toxicity = np.mean(toxicity_scores)
    print(f"Average toxicity for {model_name}: {avg_toxicity:.3f}")

    return toxicity_scores, avg_toxicity

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    full_dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    dataset = full_dataset.select(range(3000))
    
    # Use RobertaToxicityClassifier as the default scorer
    scorer = RobertaToxicityClassifier(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    results = []

    for model_name in tqdm(args.models, desc="Evaluating models"):
        print(f"\nEvaluating model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, device)

        # Create a log file for this model
        log_filename = f"{model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_filepath = os.path.join(args.save_dir, log_filename)
        
        with open(log_filepath, 'w') as log_file:
            toxicity_scores, avg_toxicity = evaluate_model(args, model, tokenizer, dataset, device, log_file, model_name, scorer)

        results.append({
            "model": model_name,
            "average_toxicity": avg_toxicity,
            "toxicity_scores": toxicity_scores
        })

        print(f"Detailed results for {model_name} logged to: {log_filepath}")

    # Write summary to CSV
    summary_filepath = os.path.join(args.save_dir, 'toxicity_results_summary_dropout.csv')
    with open(summary_filepath, 'w', newline='') as csvfile:
        fieldnames = ['model', 'average_toxicity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            row = {
                'model': result['model'],
                'average_toxicity': result['average_toxicity']
            }
            writer.writerow(row)

    print(f"Results summary written to: {summary_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output_dropout")
    parser.add_argument("--models", "-m", nargs="+", default=['llama2-7b_dropout'],
                        choices=['gpt2', 'sentence_debiasing-gender', 'sentence_debiasing-race', 'inlp-gender', 'inlp-race', 'instructive_debiasing', 
                   'self_debiasing', 'gpt2_cda_gender', 'gpt2_cda_race', 'gpt2_cda_religion', 'llama2-7b_cda_gender',
                   'llama2-7b_cda_race', 'llama2-7b_cda_religion', 'phi2_cda_gender', 'phi2_cda_race', 'phi2_cda_religion', 
                   'gpt2_dropout', 'phi2_dropout', 'llama2-7b_dropout','phi2', 'llama2-7b'])
    args = parser.parse_args()
    main(args)