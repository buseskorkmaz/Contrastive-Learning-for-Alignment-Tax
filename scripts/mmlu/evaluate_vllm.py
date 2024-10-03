import csv
import json
import argparse
import os
import os
os.environ['CURL_CA_BUNDLE'] = ''
import torch
import random
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from summarization.models import get_hf_id
from seat.debias.self_debias.modeling import GPT2Wrapper, Llama2Wrapper, Phi2Wrapper
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LlamaTokenizer, AutoTokenizer
import torch
from datetime import datetime


choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

DEBIASING_PREFIX = "Be positive, polite, respectful, compliant, loveable, modest, helpful, and kind for: "

def format_instructive_debias_prompt(prompt):
    return f"{DEBIASING_PREFIX}\n\n{prompt}"

def get_model_max_length(model_name):
    if 'phi' in model_name.lower():
        return 2047
    if 'llama' in model_name.lower():
        return 4095
    else:
        return 1023  # Default fallback

def setup_logging(args, file_prefix):
    # Ensure the log directory exists
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)

    # Create a unique log file name
    timestamp = datetime.now().strftime('%m-%d_%H-%M')
    log_file_name = f"{file_prefix}_{timestamp}_logfile.log"
    log_file_path = os.path.join(save_log_dir, log_file_name)

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(log_file_path, mode='w')
    console_handler = logging.StreamHandler(sys.stdout)

    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model(model_name):
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                        stop=["Question:"])
    print(model_name)
    if 'gpt2' in model_name:
        llm = LLM(model=model_name, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                tokenizer='gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif 'llama' in model_name:
        llm = LLM(model=model_name, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                tokenizer='meta-llama/Llama-2-7b-hf')
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", legacy=False)
    elif 'phi-2' in model_name or 'phi2' in model_name:
        llm = LLM(model=model_name, gpu_memory_utilization=float(args.gpu_util),
                tensor_parallel_size=1,
                max_model_len=max_model_length,
                trust_remote_code=True,
                tokenizer='microsoft/phi-2')
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
    return (llm, sampling_params), tokenizer


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    
    if 'instructive_debiasing' in args.model:
        prompt = format_instructive_debias_prompt(prompt)    
            
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ""
    with open(f"/gpfs/home/bsk18/factual-bias-mitigation/scripts/mmlu/cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line
    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logger.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    response_batch = []
    pred_batch = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response_batch.append(generated_text)
        pred = extract_answer(generated_text)
        pred_batch.append(pred)
    return pred_batch, response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


import logging
from tqdm import tqdm

@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    llm, sampling_params = model
    global choices
    logger.info(f"Evaluating subject: {subject}")
    inference_batches = []
    discarded_count = 0

    # Create a tqdm progress bar for the main loop
    main_pbar = tqdm(total=len(test_df), desc="Processing test data")

    for curr in test_df:
        k = args.ntrain
        prompt_length_ok = False
        prompt = None

        while not prompt_length_ok and k >= 0:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            length = len(inputs["input_ids"][0])
            
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1

        if prompt_length_ok:
            inference_batches.append(prompt)
        else:
            discarded_count += 1
            logger.info(f"Discarded a sample from subject {subject} due to length constraints")

        main_pbar.update(1)
        main_pbar.set_postfix({"Discarded": discarded_count})

    main_pbar.close()
    logger.info(f"Discarded {discarded_count} samples for subject {subject} due to length constraints")

    if not inference_batches:
        logger.warning(f"No valid samples for subject {subject} after length filtering")
        return 0.0, 0, 0, discarded_count

    logger.info(f"Starting batch inference for {len(inference_batches)} items")
    pred_batch, response_batch = batch_inference(llm, sampling_params, inference_batches)
    
    res = []
    for j, (curr, pred, response) in enumerate(zip(test_df, pred_batch, response_batch)):
        if j < len(inference_batches):  # Only process non-discarded samples
            curr["pred"] = pred
            curr["model_outputs"] = response
            res.append(curr)

    accu, corr, wrong = save_res(res, output_path)
    logger.info(f"Batch accuracy: {accu:.2f}, correct: {corr}, wrong: {wrong}")

    return accu, corr, wrong, discarded_count


def main():
    model_name = get_hf_id(args.model)
    model, tokenizer = load_model(model_name)
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    full_test_df, full_val_df = load_mmlu_pro()
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logger.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    total_discarded = 0
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        acc, corr_count, wrong_count, discarded_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)
        total_discarded += discarded_count
        logger.info(f"Discarded {discarded_count} samples for subject {subject} due to length constraints")
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}
    logger.info(f"Total discarded samples across all subjects: {total_discarded}")

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="/gpfs/home/bsk18/factual-bias-mitigation/scripts/mmlu/output/results")
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
    max_model_length = get_model_max_length(args.model)
    max_new_tokens = max_model_length // 2
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    # Setup logging
    logger = setup_logging(args, file_prefix)

    # Your existing code starts here
    logger.info("Starting the script")
    main()

