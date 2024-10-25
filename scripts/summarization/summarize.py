import os
import sys
import logging
import json
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

from src.contrastive_learning.utils import (
    print_gpu_memory, 
    read_files, 
    combine_data,
    calculate_original_factuality,
    calculate_original_toxicity,
    generate_dataset_name,
)
from src.contrastive_learning.train_utils import(
    combined_loss,
    evaluate_w_toxicity,
)
from src.contrastive_learning.model import SummarizationModel
from src.contrastive_learning.dataset import SummarizationDataset 
# from src.factuality_detector import FactualityDetector
# from src.debiasing_algorithms.sentence_debiasing.models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel 
# from src.debiasing_algorithms.inlp.models.inlp_model import INLPGPT2LMHeadModel
# from src.debiasing_algorithms.autorefine.model import AutoRefine

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from tqdm import tqdm
from transformers import AutoModel, GPT2Tokenizer, AutoModelForCausalLM
import random
random.seed(42)
import wandb
import argparse
from datetime import datetime

# Set up logging
import logging
import os
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Optionally, add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def generate_summary(model, tokenizer, text, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary


def train(model, tokenizer, train_loader, val_loader, optimizer, device, num_epochs, logger):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
            # Log training loss to WandB
            wandb.log({"train_loss": loss.item(), "epoch": epoch+1})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Average training loss: {avg_loss}")

        # Evaluate on validation set
        val_loss, val_summaries = evaluate(model, tokenizer, val_loader, device, logger, epoch)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Validation loss: {val_loss}")

        # Log validation loss to WandB
        wandb.log({"val_loss": val_loss, "epoch": epoch+1})

        # Optionally, save the model checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

def evaluate(model, tokenizer, dataloader, device, logger, epoch):
    model.eval()
    total_loss = 0
    summaries = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Generate summaries
            for idx in range(input_ids.size(0)):
                source_text = batch['source_text'][idx]
                target_text = batch['target_text'][idx]

                # Prepare input_ids for generation (source text only)
                source_encoding = tokenizer(
                    source_text,
                    return_tensors='pt',
                    max_length=512,          # Limit the input length
                    truncation=True,
                    padding='max_length'
                ).to(device)

                generated_ids = model.generate(
                    input_ids=source_encoding['input_ids'],
                    attention_mask=source_encoding['attention_mask'],
                    max_new_tokens=50,      # Generate up to 50 new tokens
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set
                )
                generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                summaries.append((source_text, generated_summary, target_text))
                # print(f"Source text: {source_text}")
                # print(f"generate_summary: {generated_summary}")
                # print(f"Original summary: {target_text}")

                logger.info(f"Source text: {source_text}\n")
                logger.info(f"generate_summary: {generated_summary}\n")
                logger.info(f"Original summary: {target_text}\n")

                # Log summaries to WandB
                if idx < 10:  # Limit the number of samples logged per epoch
                    wandb.log({
                        f"epoch_{epoch+1}_val_sample_{idx+1}_source": source_text,
                        f"epoch_{epoch+1}_val_sample_{idx+1}_generated": generated_summary,
                        f"epoch_{epoch+1}_val_sample_{idx+1}_reference": target_text,
                        "epoch": epoch+1
                    })

    avg_loss = total_loss / len(dataloader)
    return avg_loss, summaries


if __name__ == "__main__":
    logger.info("Starting main function...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_data', nargs='+', choices=['original', 'bt', 'all'], default=['original'], help='Positive data options')
    args = parser.parse_args()

    # Process 'all' option
    all_pos_options = ['original', 'bt']    
    if 'all' in args.pos_data:
        args.pos_data = all_pos_options

    # Ensure 'original' is always included
    if 'original' not in args.pos_data:
        args.pos_data.insert(0, 'original')

    # Define data paths based on arguments
    pos_data_paths = ['/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/pos']

    for opt in args.pos_data:
        if opt != 'original':
            pos_data_paths.append(f'/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/pos_{opt}')

    # Combine data from all specified paths
    pos_data = combine_data([read_files(path, logger) for path in pos_data_paths])

   # Initialize the model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained('gpt2')

    # Set pad_token_id in model configuration
    model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Initialized model: {model_name}")

    # Initialize datasets
    train_dataset = SummarizationDataset(pos_data['train'], tokenizer, max_length=512)
    val_dataset = SummarizationDataset(pos_data['validation'], tokenizer, max_length=512)
    test_dataset = SummarizationDataset(pos_data['test'], tokenizer, max_length=512)

    # Create config dictionary
    config = {
        "ce_weight": 1,
        "model_name": model_name,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "epochs": 10,
        "pos_data_option": args.pos_data,
    }
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    logger.info("Created datasets")

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    # Initialize WandB
    wandb.init(
        project='factual-bias-mitigation-scripts_tldr_train',
        name=f"{model_name}-{'_'.join(args.pos_data)}",
    )

    wandb.config.update({
        "model_name": model_name,
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
        "pos_data_option": args.pos_data,
    })

    # Start training
    train(
        model, 
        tokenizer, 
        train_loader, 
        val_loader, 
        optimizer, 
        device,
        num_epochs=config["epochs"],
        logger=logger
    )

    # Evaluate on test set
    test_loss, test_summaries = evaluate(model, tokenizer, test_loader, device, logger, epoch='test')
    logger.info(f"Test loss: {test_loss}")
    wandb.log({"test_loss": test_loss})

    # Log test summaries
    for idx, (source_text, generated_summary, reference_summary) in enumerate(test_summaries):
        logger.info(f"Test Sample {idx+1}")
        logger.info(f"Source: {source_text}")
        logger.info(f"Generated Summary: {generated_summary}")
        logger.info(f"Reference Summary: {reference_summary}")
        logger.info("---")

        # Log to WandB
        if idx < 5:  # Limit the number of samples logged
            wandb.log({
                f"test_sample_{idx+1}_source": source_text,
                f"test_sample_{idx+1}_generated": generated_summary,
                f"test_sample_{idx+1}_reference": reference_summary,
                "epoch": 'test'
            })

    # Finish WandB run
    wandb.finish()