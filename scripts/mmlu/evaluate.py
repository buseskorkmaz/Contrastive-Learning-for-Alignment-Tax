import argparse
import os
import sys
import json
import time
import csv
import random
import logging
import torch
import re
from tqdm import tqdm
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from scripts.summarization.models import init_debiased_model
from scripts.seat.debias.self_debias.modeling import GPT2Wrapper, Llama2Wrapper, Phi2Wrapper
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def get_model_max_length(model, tokenizer):
    if isinstance(model, GPT2LMHeadModel) or isinstance(tokenizer, GPT2Tokenizer):
        return 1024
    elif hasattr(model.config, 'max_position_embeddings'):
        return model.config.max_position_embeddings
    else:
        return 4096  # Default fallback


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048

DEBIASING_PREFIX = "Be positive, polite, respectful, compliant, loveable, modest, helpful, and kind for: "

def format_instructive_debias_prompt(prompt):
    return f"{DEBIASING_PREFIX}\n\n{prompt}"

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = [opt for opt in each["options"] if opt != "N/A"]
        each["options"] = options
        res_df.append(each)
    return res_df

def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]

def select_by_category(df, subject):
    return [each for each in df if each["category"] == subject]

def format_cot_example(example, including_answer=True):
    prompt = f"Question:\n{example['question']}\nOptions:\n"
    prompt += "\n".join([f"{choices[i]}. {opt}" for i, opt in enumerate(example["options"])])
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        prompt += f"\n{cot_content}\n\n"
    else:
        prompt += "\nAnswer: Let's think step by step."
    return prompt

def generate_cot_prompt(val_df, curr, k, model_name):
    logging.info(f"Generating prompt with k={k}")
    with open(f"/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/mmlu/cot_prompt_lib/initial_prompt.txt", "r") as fi:
        prompt = fi.read().replace("{$}", curr["category"]) + "\n"
    
    val_df = select_by_category(val_df, curr["category"])[:k]
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    
    if "instructive_debiasing" in model_name:
        prompt = format_instructive_debias_prompt(prompt)
    
    logging.info(f"Generated prompt length: {len(prompt)}")
    return prompt

def extract_answer(text):
    patterns = [
        r"answer is \(?([A-J])\)?",
        r'.*[aA]nswer:\s*([A-J])',
        r"\b[A-J]\b(?!.*\b[A-J]\b)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def batch_inference(model, tokenizer, inference_batch, model_name):
    device = next(model.parameters()).device
    outputs = []
    
    max_length = get_model_max_length(model, tokenizer)
    max_new_tokens = min(2048, max_length // 2)  # Adjust max_new_tokens based on model's max length
    
    for prompt in tqdm(inference_batch, desc="Batch inference"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        if isinstance(model, (GPT2Wrapper, Llama2Wrapper, Phi2Wrapper)):
            self_debiasing_prefixes = [
                "The following text contains rude, disrespectful, or unreasonable language:\n",
                "The following text contains very hateful, aggressive, disrespectful language:\n",
                "The following text contains sexually explicit language:\n",
                "The following text contains a threat:\n",
                "The following text contains negative or hateful language targeting someone because of their identity:\n",
                "The following text contains swear words, curse words, or other obscene or profane language:\n"
            ]
            output = model.generate_self_debiasing(
                input_texts=[prompt],
                debiasing_prefixes=self_debiasing_prefixes,
                max_length=min(input_ids.shape[-1] + max_new_tokens, max_length),
                num_return_sequences=1,
            )
            generated_text = output
        else:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=min(input_ids.shape[-1] + max_new_tokens, max_length),
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True
            )
            generated_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        outputs.append(generated_text)
    
    return outputs

def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        json.dump(res, fo, indent=2)
    
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    
    total = corr + wrong
    accu = corr / total if total > 0 else 0.0
    return accu, corr, wrong

@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, model_name):
    logging.info(f"Evaluating {subject}")
    inference_batches = []

    max_length = get_model_max_length(model, tokenizer)
    max_new_tokens = min(2048, max_length // 2)

    for curr_idx, curr in enumerate(tqdm(test_df, desc=f"Preparing prompts for {subject}")):
        k = args.ntrain
        prompt_length_ok = False
        attempts = 0
        while not prompt_length_ok and attempts < 10:  # Add a maximum number of attempts
            prompt = generate_cot_prompt(val_df, curr, k, model_name)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            prompt_length = len(inputs["input_ids"][0])
            logging.info(f"Prompt {curr_idx}: length = {prompt_length}, k = {k}")
            if prompt_length < max_length - max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1
                attempts += 1

        if not prompt_length_ok:
            logging.warning(f"Could not create a short enough prompt for item {curr_idx}. Using the last attempt.")

        inference_batches.append(prompt)

        # Log progress every 50 items
        if curr_idx % 50 == 0:
            logging.info(f"Processed {curr_idx} / {len(test_df)} items")

    logging.info("Starting batch inference")
    response_batch = batch_inference(model, tokenizer, inference_batches, model_name)
    logging.info("Finished batch inference")
    
    res = []
    for curr, response in zip(test_df, response_batch):
        pred = extract_answer(response)
        curr["pred"] = pred
        curr["model_outputs"] = response
        res.append(curr)

    accu, corr, wrong = save_res(res, output_path)
    logging.info(f"Accuracy for {subject}: {accu:.4f}, correct: {corr}, wrong: {wrong}")
    return accu, corr, wrong

def main():
    model, tokenizer = init_debiased_model(args.model)
    max_length = get_model_max_length(model, tokenizer)
    logging.info(f"Using max_length of {max_length} for model {args.model}")

    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = list(set(each["category"] for each in full_test_df))
    
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = [sub for sub in all_subjects if any(each.replace(" ", "_") in sub.replace(" ", "_") for each in args.selected_subjects.split(","))]
    
    logging.info("Selected subjects:\n" + "\n".join(selected_subjects))
    print("Selected subjects:\n" + "\n".join(selected_subjects))
    
    sta_dict = {subject: {"corr": 0.0, "wrong": 0.0, "accu": 0.0} for subject in selected_subjects}
    
    with open(summary_path, 'a') as f:
        f.write("\n------category level sta------\n")
    
    for subject in sorted(selected_subjects):
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, f"{subject}.json")
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path, args.model)
        sta_dict[subject] = {"corr": corr_count, "wrong": wrong_count, "accu": acc}
        
        with open(summary_path, 'a') as f:
            f.write(f"Average accuracy {acc:.4f} - {subject}\n")
    
    total_corr = sum(v["corr"] for v in sta_dict.values())
    total_wrong = sum(v["wrong"] for v in sta_dict.values())
    total_accu = total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0

    with open(summary_path, 'a') as f:
        f.write("\n------average acc sta------\n")
        f.write(f"Average accuracy: {total_accu:.4f}\n")
    
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, total_accu]
        writer.writerow(record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/mmlu/output/results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()