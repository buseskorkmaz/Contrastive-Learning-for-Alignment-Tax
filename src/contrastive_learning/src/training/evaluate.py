from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import pipeline
# from src.utils.factuality_detector import FactualityDetector

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import tqdm

import numpy as np

from torch import nn
from nltk import sent_tokenize
import itertools
import spacy

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel
import pandas as pd
import logging
from tqdm import tqdm
from datasets import load_dataset
import torch
from transformers import pipeline
from accelerate import Accelerator

class FactualityDetector:
    def __init__(self, modelname, max_seq_length=512):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained('factual-bias-mitigation/factuality_detector_model', torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained('factual-bias-mitigation/factuality_detector_model')
        self.max_seq_length = max_seq_length

        self.model = self.model.to(self.device)
        self.model.eval()
        self.nlp = spacy.load("en_core_web_sm")

    def break_context(self, context, n_gram=1, n_skip=1, all_possible=False):
        sents = sent_tokenize(context)
        len_nums = len(sents)
        sent_fragments = []
        if all_possible:
            sent_fragments = [" ".join(x) for x in list(itertools.permutations(sents, n_gram))]
        else:
            for k in range(0, len_nums, n_skip):
                sent_fragments.append(" ".join(sents[k:k+n_gram]))
        return sent_fragments
    
    
    def compute_summac_score(self, raw_batch, pool_mode='avg', return_matrix=False, n_gram=2, n_skip=2, all_possible=False):
        batch_response = []
        batch_value_matrix = []
        batch_context_response = []
        for item in raw_batch:
            context, response = item
            context_sents = self.break_context(context, n_gram=n_gram, n_skip=n_skip, all_possible=all_possible)
            response_sents = self.break_context(response)
            M=len(context_sents)
            N=len(response_sents)
            value_matrix = np.zeros(shape=(M, N))
            batch_data = []
            for this_context in (context_sents):
                for this_response in (response_sents):
                    batch_data.append((this_context, this_response))
            # [(this_context, this_response)]
            # import ipdb; ipdb.set_trace()
            model_input = self.tokenizer(batch_data, padding=True, truncation=True, 
                                         max_length=self.max_seq_length, return_tensors="pt")
            model_input = model_input.to(self.device)
            with torch.no_grad(): 
                logits = self.model(**model_input).logits
            probabilities = nn.functional.softmax(logits, dim=-1)
            generated_rewards=probabilities[:,1].detach().cpu().numpy()
            value_matrix=np.reshape(generated_rewards, (M,N)) # generated_rewards.item()

            batch_value_matrix.append(value_matrix)
            batch_context_response.append((context_sents, response_sents))
            response_factuality = np.max(value_matrix, axis=0)
            response_factuality = np.mean(response_factuality) if pool_mode=='avg' else np.min(response_factuality)
            batch_response.append(response_factuality)
    
        if return_matrix:
            return batch_value_matrix, batch_context_response, batch_response
        else:
            return batch_response


    def is_single_fact_response(self, input_text):
        # Process the input text with spaCy
        doc = self.nlp(input_text)
        # Check for the presence of a subject and a verb
        is_single_fact = all([token.pos_ in ['PROPN', 'ADP', 'PROPN'] for token in doc])
        return is_single_fact

    def compute_direct_scores(self, raw_batch):
        model_input = self.tokenizer(raw_batch, padding=True, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        model_input = model_input.to(self.device)
        with torch.no_grad(): 
            logits = self.model(**model_input).logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        generated_rewards=probabilities[:,1].cpu()
        return generated_rewards

    def generate_score(self, contexts, responses, summac_style=False, n_gram=2, n_skip=2, 
                       all_possible=False, compare_with_direct_score=True, direct_compare_threshold=0.3,
                       add_question_if_not_sentence=False, questions=None):
        for k, r in enumerate(responses):
            if self.is_single_fact_response(r) and add_question_if_not_sentence:
                responses[k] = f"{questions[k]} {r}"
        # print(responses)
                
        raw_batch = [(c, r) for c, r in zip(contexts, responses)]
        if summac_style:
            value_matrix, context_response, factuality_score = self.compute_summac_score(raw_batch, return_matrix=True, 
                                                                                         n_gram=n_gram, n_skip=n_skip, 
                                                                                         all_possible=all_possible)
            # only use direct baseline if the summac style factuality score is not close to 0 or 1
            threshold_condition = (factuality_score[0]>direct_compare_threshold) \
                and (factuality_score[0]<(1-direct_compare_threshold))
            if compare_with_direct_score and threshold_condition:
                direct_score = self.compute_direct_scores(raw_batch).item()
                factuality_score = max(factuality_score[0], direct_score)
            else:
                factuality_score = factuality_score[0]
            return value_matrix, context_response, factuality_score
        else:
            generated_rewards = self.compute_direct_scores(raw_batch)
            return generated_rewards

    def generate_score_auto_ngram(self, contexts, responses, summac_style=False, 
                                  n_gram_list=[2, 3, 4, 5, 6], n_skip=2, all_possible=False):
        raw_batch = [(c, r) for c, r in zip(contexts, responses)]
        assert summac_style, "SummaC needs to be true for auto ngram"
        best_score=0
        best_ngram=0
        best_returns=None
        for n_gram in n_gram_list:
            value_matrix, context_response, factuality_score = self.compute_summac_score(raw_batch, return_matrix=True, 
                                                                                         n_gram=n_gram, n_skip=n_skip, 
                                                                                         all_possible=all_possible,
                                                                                         compare_with_direct_score=False)
            if factuality_score[0]>best_score:
                best_score=factuality_score[0]
                best_ngram=n_gram
                best_returns=(value_matrix, context_response, factuality_score)

        print("Best ngram: ", best_ngram)
        return best_returns


def evaluate_test_metrics(model, tokenizer, accelerator, logger, state):
    """
    Evaluates model performance on test data using toxicity and faithfulness metrics.
    This matches the original evaluation setup, focusing purely on inference.
    """
    model.eval()
    test_metrics = {}

    try:
        # Load dataset in the same way as original evaluation
        dataset = load_dataset('dim/tldr_17_3k', split="train[:4%]")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return test_metrics

    print("Generating summaries for evaluation...\n", flush=True)
    
    all_sources = []
    all_generated = []
    
    # Process each example individually, matching inference setup
    for example in tqdm(dataset, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            # Prepare input just like in inference
            inputs = tokenizer(
                example['content'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(accelerator.device)
            
            # Generate summary
            generated_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=64,
                repetition_penalty=1.0,
            )
            
            # Remove prompt from generation
            prompt_len = inputs['input_ids'].size(1)
            generated_ids = generated_ids[:, prompt_len:]
            
            # Decode the generation
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            all_sources.append(example['content'])
            all_generated.append(generated_text)
            # print("Generated Text:", generated_text, flush=True)
            # print(len(all_generated), flush=True)
            # print(len(all_sources), flush=True)

    # Compute metrics on main process
    if accelerator.is_main_process:
        # Initialize evaluation models
        toxicity_model = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        factuality_detector = FactualityDetector(
            "/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli"
        )

        # Compute toxicity scores
        print("Computing toxicity scores...", flush=True)
        toxicity_results = toxicity_model(all_generated, batch_size=32, truncation=True, max_length=512)
        toxicity_scores = [result['score'] for result in toxicity_results]

        # Compute faithfulness scores
        print("Computing faithfulness scores...", flush=True)
        errored_samples = 0
        faithfulness_scores = []
        for source, generated in zip(all_sources, all_generated):
            if generated == "":
                faithfulness_scores.append(None)
                continue

            try:
                _, _, factuality_score = factuality_detector.generate_score(
                    [source], [generated], 
                    summac_style=True
                )
            except Exception as e:
                errored_samples +=1
                logger.error(f"Error in factuality detection: {str(e)}")
                logger.info("errored sample: ", errored_samples, generated)
                factuality_score = None

            faithfulness_scores.append(factuality_score)
            
        print("Faithfulness scores: ", faithfulness_scores)
        # Calculate average scores
        avg_toxicity = sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0
        valid_faithfulness = [s for s in faithfulness_scores if s is not None]
        avg_faithfulness = sum(valid_faithfulness) / len(valid_faithfulness) if valid_faithfulness else 0

        # Store metrics
        test_metrics.update({
            "test/toxicity": avg_toxicity,
            "test/faithfulness": avg_faithfulness,
            "test/toxicity_std": torch.tensor(toxicity_scores).std().item(),
            "test/faithfulness_std": torch.tensor([s for s in faithfulness_scores if s is not None]).std().item()
        })

        # Log example outputs
        log_filename = f"test_generation_outputs_epoch_{state['epoch']}.txt"
        with open(log_filename, 'w', encoding='utf-8') as f:
            for idx in range(min(5, len(all_sources))):
                f.write(f"Example {idx + 1}\n")
                f.write(f"Source: {all_sources[idx]}\n")
                f.write(f"Generated: {all_generated[idx]}\n")
                f.write(f"Toxicity: {toxicity_scores[idx]:.3f}\n")
                f.write(f"Faithfulness: {faithfulness_scores[idx]:.3f}\n")
                f.write("-" * 80 + "\n")

    accelerator.wait_for_everyone()
    return test_metrics

# def evaluate_after_training():
#     from transformers import AutoTokenizer, AutoModelForCausalLM
#     import pandas as pd
#     import logging
#     from tqdm import tqdm
#     from datasets import load_dataset
#     import torch
#     from transformers import pipeline

#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)

#     tokenizer = AutoTokenizer.from_pretrained("constrastove_training/output/models/gpt2-alpha-2/checkpoint-step-276")
#     base_model = AutoModelForCausalLM.from_pretrained("constrastove_training/output/models/gpt2-alpha-2/checkpoint-step-276", device_map='auto')
#     base_model.eval()

#     try:
#         dataset = load_dataset('dim/tldr_17_3k')["train"]
#     except Exception as e:
#         print(f"Failed to load dataset: {str(e)}")
#         return {}

#     results = {
#         "content": [],
#         "original_summary": [],
#         "generated_summary": [],
#     }

#     # Generate summaries
#     logger.info("Generating summaries for evaluation...")
#     for example in tqdm(dataset):
#         source_text = example['content']
#         original_summary = example['summary']

#         inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=512).to(base_model.device)

#         with torch.no_grad():
#             generated_ids = base_model.generate(
#                 input_ids=inputs['input_ids'],
#                 attention_mask=inputs['attention_mask'],
#                 max_new_tokens=64,
#                 repetition_penalty=1.0,
#             )

#         prompt_len = inputs['input_ids'].size(1)
#         gen_ids = generated_ids[:, prompt_len:]
#         generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

#         results["content"].append(source_text)
#         results["original_summary"].append(original_summary)
#         results["generated_summary"].append(generated_text)

#     # Save partial CSV (without scores)
#     df_no_scores = pd.DataFrame(results)
#     os.makedirs("constrastove_training/output/evaluation/", exist_ok=True)
#     df_no_scores.to_csv(
#         "constrastove_training/output/evaluation/gpt2-alpha-2_generations_no_scores.csv", 
#         index=False
#     )
#     logger.info("Saved CSV without toxicity/faithfulness scores.")

#     # Evaluate toxicity
#     logger.info("Evaluating toxicity...")
#     toxicity_model = pipeline("text-classification", model="unitary/toxic-bert", device=0 if torch.cuda.is_available() else -1)
#     generated_toxicity = toxicity_model(results["generated_summary"], batch_size=32, truncation=True, max_length=512)
#     results["generated_toxicity"] = [res['score'] for res in generated_toxicity]

#     # Evaluate faithfulness
#     logger.info("Evaluating faithfulness...")
#     factuality_detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")
#     faithfulness_scores = []
#     for src, gen in zip(results["content"], results["generated_summary"]):
#         if not gen.strip():
#             faithfulness_scores.append(None)
#             continue
#         try:
#             _, _, fscore = factuality_detector.generate_score([src], [gen], summac_style=True)
#             faithfulness_scores.append(fscore)
#         except Exception as e:
#             faithfulness_scores.append(None)
#     results["faithfulness"] = faithfulness_scores

#     # Final CSV with scores
#     df_with_scores = pd.DataFrame(results)
#     df_with_scores.to_csv(
#         "constrastove_training/output/evaluation/gpt2-alpha-2_generation_with_scores.csv",
#         index=False
#     )
#     logger.info("Final results saved to gpt2-alpha-2_results.csv")

#     # Optionally compute summary stats if desired:
#     avg_tox = sum(results["generated_toxicity"]) / len(results["generated_toxicity"])
#     valid_faithful = [s for s in faithfulness_scores if s is not None]
#     avg_faithful = sum(valid_faithful) / len(valid_faithful) if valid_faithful else 0

#     print(f"Average Toxicity: {avg_tox:.3f}, Average Faithfulness: {avg_faithful:.3f}")
#     return {"test/toxicity": avg_tox, "test/faithfulness": avg_faithful}

# evaluate_after_training()

def evaluate_after_training():
    accelerator = Accelerator()
    # Only display tqdm on the local main process
    show_tqdm = accelerator.is_local_main_process
    
    # -----------------------
    # 1) Load model & tokenizer
    # -----------------------
    print("Loading model and tokenizer...")
    checkpoint_dir = "constrastove_training/output/training/gpt2/alpha-2-final_checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     checkpoint_dir,
    #     # device_map="auto"  # lets Accelerate place layers across all GPUs
    # )
    # base_model.eval()
    # base_model = accelerator.prepare(base_model)

    # Load the raw checkpoint
    ckpt_sd = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location="cpu")

    # Create the same model architecture as your original
    config = GPT2Config.from_pretrained("gpt2")  # or whichever config
    base_model = GPT2LMHeadModel(config)

    # The model's official keys:
    model_sd = base_model.state_dict()

    # Prepare a dict for matching keys
    fixed_sd = {}

    for ck, cv in ckpt_sd.items():
        # If your old keys had a prefix like 'base_model.', remove it:
        new_key = ck.replace("base_model.", "")
        # If that matches a key in the model, keep it:
        if new_key in model_sd:
            fixed_sd[new_key] = cv

    # Now load, allowing missing or extra keys
    base_model.load_state_dict(fixed_sd, strict=False)
    base_model.eval()
    base_model = accelerator.prepare(base_model)

    # -----------------------
    # 2) Load and shard dataset
    # -----------------------
    try:
        # Use the same dataset you had before
        dataset = load_dataset('dim/tldr_17_3k')["train"]
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return {}

    # Shard the dataset so each GPU handles a portion
    # If you have 4 GPUs, each rank processes ~1/4 of the dataset.
    dataset_shard = dataset.shard(
        num_shards=accelerator.num_processes,
        index=accelerator.process_index
    )

    # -----------------------
    # 3) Generate partial results
    # -----------------------
    print("Generating summaries on each shard...")
    local_results = {"content": [], "original_summary": [], "generated_summary": []}

    for example in tqdm(dataset_shard, disable=not show_tqdm):
        source_text = example["content"]
        original_summary = example["summary"]

        inputs = tokenizer(
            source_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        # Move inputs to correct device for this rank
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = base_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                repetition_penalty=1.0,
            )

        # Remove the prompt
        prompt_len = inputs["input_ids"].size(1)
        gen_ids = generated_ids[:, prompt_len:]
        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        print(generated_text[:60])
        local_results["content"].append(source_text)
        local_results["original_summary"].append(original_summary)
        local_results["generated_summary"].append(generated_text)

    # Save partial CSV with no scores (just for debugging or inspection)
    partial_dir = "constrastove_training/output/evaluation"
    os.makedirs(partial_dir, exist_ok=True)
    
    partial_csv_path = os.path.join(partial_dir, f"partial_rank_{accelerator.process_index}.csv")
    pd.DataFrame(local_results).to_csv(partial_csv_path, index=False)
    
    accelerator.wait_for_everyone()

    # -----------------------
    # 4) Merge partial CSVs on rank=0
    # -----------------------
    full_results = {"content": [], "original_summary": [], "generated_summary": []}

    if accelerator.is_main_process:
        # Gather all partial CSVs into one DataFrame
        print("Merging partial results on rank=0...")
        all_partials = []
        for rank_id in range(accelerator.num_processes):
            rank_csv = os.path.join(partial_dir, f"partial_rank_{rank_id}.csv")
            df_part = pd.read_csv(rank_csv)
            all_partials.append(df_part)
        df_merged = pd.concat(all_partials, ignore_index=True)
        
        # If you want to shuffle or keep the order, do so here.
        # df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save CSV before scores
        no_scores_path = os.path.join(partial_dir, "gpt2-alpha-2_generations_no_scores.csv")
        df_merged.to_csv(no_scores_path, index=False)
        print(f"Saved CSV without toxicity/faithfulness scores to: {no_scores_path}")

        # -----------------------
        # 5) Evaluate Toxicity (still on rank=0)
        # -----------------------
        print("Evaluating toxicity on rank=0...")
        toxicity_model = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=0 if torch.cuda.is_available() else -1
        )
        # The generated_summary column
        df_merged["generated_summary"] = df_merged["generated_summary"].fillna("")
        df_merged["generated_summary"] = df_merged["generated_summary"].astype(str)
        generated_texts = df_merged["generated_summary"].tolist()
        generated_toxicity = toxicity_model(
            generated_texts,
            batch_size=32,
            truncation=True,
            max_length=512
        )
        df_merged["generated_toxicity"] = [res["score"] for res in generated_toxicity]

        # -----------------------
        # 6) Evaluate Faithfulness
        # -----------------------
        print("Evaluating faithfulness on rank=0...")
        factuality_detector = FactualityDetector(
            "/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli"
        )

        faithfulness_scores = []
        for src, gen in zip(df_merged["content"], df_merged["generated_summary"]):
            gen = gen.strip()
            if not gen:
                faithfulness_scores.append(None)
                continue
            try:
                _, _, fscore = factuality_detector.generate_score([src], [gen], summac_style=True)
                faithfulness_scores.append(fscore)
            except Exception:
                faithfulness_scores.append(None)
        df_merged["faithfulness"] = faithfulness_scores

        # -----------------------
        # 7) Save final CSV
        # -----------------------
        final_path = os.path.join(partial_dir, "gpt2-alpha-2_generation_with_scores.csv")
        df_merged.to_csv(final_path, index=False)
        print(f"Final results with scores saved to: {final_path}")

        # Optionally compute summary stats
        avg_toxicity = df_merged["generated_toxicity"].mean()
        valid_faith = df_merged["faithfulness"].dropna()
        avg_faithfulness = valid_faith.mean() if len(valid_faith) > 0 else 0.0

        print(f"Average Toxicity: {avg_toxicity:.3f}, Average Faithfulness: {avg_faithfulness:.3f}")
        
        # Return a dict of final metrics
        return {
            "test/toxicity": float(avg_toxicity),
            "test/faithfulness": float(avg_faithfulness)
        }

    else:
        # Non-main ranks return nothing special
        return {}

if __name__ == "__main__":
    evaluate_after_training()



