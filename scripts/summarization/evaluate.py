import os
import sys
import pandas as pd
from tqdm import tqdm
import logging
from datasets import load_dataset
import torch
from transformers import pipeline, GPT2Tokenizer, AutoModelForCausalLM
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from seat.debias.self_debias.modeling import GPT2Wrapper, Llama2Wrapper, Phi2Wrapper
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from src.factuality_detector import FactualityDetector
import traceback
import argparse
from models import init_debiased_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summaries(model, tokenizer, content, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEBIASING_PREFIX = "Be positive, polite, respectful, compliant, loveable, modest, helpful, and kind for: "
    sd_debiasing_prefixes = [
        "The following text contains rude, disrespectful, or unreasonable language:\n",
        "The following text contains very hateful, aggressive, disrespectful language:\n",
        "The following text contains sexually explicit language:\n",
        "The following text contains a threat:\n",
        "The following text contains negative or hateful language targeting someone because of their identity:\n",
        "The following text contains swear words, curse words, or other obscene or profane language:\n"
    ]
    
    if "instructive_debiasing" in model_name:
        content = DEBIASING_PREFIX + ' ' + content 

    input_ids = tokenizer(content, return_tensors="pt", truncation=True).input_ids.to(device)
    max_length = 1024 - 128 - 1
    input_ids = input_ids[:, :max_length]
    max_len = input_ids.shape[-1] + 128       

    if isinstance(model, (GPT2Wrapper, Llama2Wrapper, Phi2Wrapper)): 
        if isinstance(model, GPT2Wrapper):
            max_len=len(tokenizer.encode(content, truncation=True, max_length=max_length-512)) + 128
       
        outputs = model.generate_self_debiasing(
                input_texts=[content],
                debiasing_prefixes=sd_debiasing_prefixes,
                max_length=max_len,
                num_return_sequences=1,
                top_k=1
            )
    else:    
        outputs = model.generate(input_ids, top_k=1, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True)
        
        sequences, scores = outputs.sequences, outputs.scores

        gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
        gen_sequences = gen_sequences.cpu().numpy()

        output_str = tokenizer.decode(gen_sequences, skip_special_tokens=True)
    
    output_str = output_str.strip()
    return output_str

def prepare_dataset():
    dataset = load_dataset('dim/tldr_17_3k')["train"]
    return dataset

def evaluate_toxicity(texts):
    toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", device=0 if torch.cuda.is_available() else -1)
    results = toxicity_model(texts, batch_size=32, truncation=True, max_length=512)
    return [result['score'] for result in results]

def evaluate_faithfulness(original_summaries, generated_summaries):
    factuality_detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")
    factuality_scores = []
    for source, target in zip(original_summaries, generated_summaries):
        if source is None or target is None:
            factuality_scores.append(None)
            continue
        try:
            _, _, factuality_score = factuality_detector.generate_score([source], [target], summac_style=True)
            factuality_scores.append(factuality_score)
        except IndexError:
            print(f"IndexError in SummaC for source: {source[:50]}... and target: {target[:50]}...")
            factuality_scores.append(None)
        except Exception as e:
            print(f"Error in evaluate_faithfulness: {str(e)}")
            print(f"Source: {source}")
            print(f"Target: {target}")
            factuality_scores.append(None)
    return factuality_scores

def main():
    dataset = prepare_dataset()
    # model_names = ["sentence_debiasing", "inlp", "instructive_debiasing", "self_debiasing", "original", "gpt2-tldr"]
    # model_names = ["self_debiasing-phi2", "self_debiasing-llama2", "self_debiasing-gpt2", "instructive_debiasing-phi2",
    #                "instructive_debiasing-llama2", "instructive_debiasing-gpt2", "gpt2", "llama2-7b", "phi2", "gpt2_cda_race",
    #                "gpt2_cda_gender", "gpt2_cda_religion", "llama2-7b_cda_race", "llama2-7b_cda_gender", "llama2-7b_cda_religion",
    #                "gpt2_dropout", "llama2-7b_dropout", "phi2_dropout", "inlp-race", "inlp-gender", "sentence_debiasing-race",
    #                "sentence_debiasing-gender"]

    parser = argparse.ArgumentParser(description="Generate summaries using a specified model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for generation")
    args = parser.parse_args()

    model_name = args.model_name
    dataset = prepare_dataset()
    results = {model_name: {'content': [], 'original_summary': [], 'generated_summary': []}}

    logger.info(f"Processing model: {model_name}")
    try:
        if model_name != "original":
            model, tokenizer = init_debiased_model(model_name)
        else:
            model, tokenizer = None, None

        for i, item in tqdm(enumerate(dataset), total=len(dataset), desc=f"Processing {model_name}"):
            content = item['content']
            original_summary = item['summary']

            if model_name != "original":
                generated_summary = generate_summaries(model, tokenizer, content, model_name)
            else:
                generated_summary = original_summary

            results[model_name]['content'].append(content)
            results[model_name]['original_summary'].append(original_summary)
            results[model_name]['generated_summary'].append(generated_summary)

            if i % 10 == 0:
                logger.info(f"Sample {i} - Original Summary: {original_summary[:100]}...")
                logger.info(f"Sample {i} - Generated Summary: {generated_summary[:100]}...")

    except Exception as e:
        logger.info(f"Error processing model {model_name}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        traceback.print_exc()

    df = pd.DataFrame(results[model_name])
    df.to_csv(f"your-path/factual-bias-mitigation/scripts/summarization/output/{model_name}_generations_no_scores.csv", index=False)

    # Evaluate toxicity
    logger.info(f"Evaluating toxicity for model: {model_name}")
    original_toxicity = evaluate_toxicity(results[model_name]['original_summary'])
    generated_toxicity = evaluate_toxicity(results[model_name]['generated_summary'])
    results[model_name]['original_toxicity'] = original_toxicity
    results[model_name]['generated_toxicity'] = generated_toxicity

    # Evaluate faithfulness
    logger.info(f"Evaluating faithfulness for model: {model_name}")
    faithfulness_scores = evaluate_faithfulness(results[model_name]['original_summary'], results[model_name]['generated_summary'])
    results[model_name]['faithfulness'] = faithfulness_scores

    # Save results
    df = pd.DataFrame(results[model_name])
    df.to_csv(f"your-path/factual-bias-mitigation/scripts/summarization/output/{model_name}_results.csv", index=False)
    logger.info(f"Results for {model_name} saved to {model_name}_results.csv")

    logger.info("Evaluation complete. Results saved to CSV file.")

if __name__ == "__main__":
    main()