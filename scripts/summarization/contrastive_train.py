import os
import sys
import logging
import json
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb

from transformers import GPT2Tokenizer, GPT2Config
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

# Assume these are defined in your project based on previous code
from src.contrastive_learning.contrastive_gpt2_architecture import ContrastiveGPT2Model  # Our adjusted model
from src.contrastive_learning.contrastive_dataset import ContrastiveTranslationDataset, contrastive_collate_fn  # Our dataset and collate function
from src.contrastive_learning.contrastive_loss import ContrastiveLoss  # Our custom loss function
from src.factuality_detector import FactualityDetector
from src.contrastive_learning.train_utils import (
    evaluate_toxicity  # Function to compute toxicity scores,
)
from src.contrastive_learning.utils import( 
    calculate_original_factuality,
    calculate_original_toxicity,
    load_indices,
    combine_data,
    read_files,
)

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

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

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def train(model, tokenizer, train_loader, val_loader, optimizer, device, config, epochs, factuality_detector, original_val_factuality, original_val_toxicity):
    logger.info(f"Original Validation Factuality Score: {original_val_factuality:.4f}")
    logger.info("Starting training...")
    criterion = ContrastiveLoss(
        alpha=config["contrastive_weight"],
        tau=config["tau"],
        padding_idx=tokenizer.pad_token_id
    )
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_contrastive_loss = 0
        total_train_ce_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
        
            # Move tensors to device
            input_ids = batch['contrast_input_ids'].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            target_ids = batch['contrast_target_input_ids'].to(device)
            contrast_ne_masks = batch['contrast_ne_masks'].to(device)
            valid_contrast = batch['valid_contrast'].to(device)
            positive_contrast = batch['positive_contrast'].to(device)
            sample = {
                'contrast_ne': contrast_ne_masks,
                'valid_contrast': valid_contrast,
                'positive_contrast': positive_contrast
            }
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            lm_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            loss, lm_loss, contrastive_loss = criterion(
                lm_logits=lm_logits,
                target_ids=target_ids,
                hidden_states=hidden_states,
                sample=sample
            )
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_contrastive_loss += contrastive_loss.item()
            total_train_ce_loss += lm_loss.item()
            torch.cuda.empty_cache()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_contrastive_loss = total_train_contrastive_loss / len(train_loader)
        avg_train_ce_loss = total_train_ce_loss / len(train_loader)
        
        # Validation
        val_loss, val_contrastive_loss, val_ce_loss, val_factuality, val_toxicity = evaluate_w_toxicity(
            model, tokenizer, val_loader, device, factuality_detector, logger, criterion
        )
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_contrastive_loss": avg_train_contrastive_loss,
            "train_ce_loss": avg_train_ce_loss,
            "val_loss": val_loss,
            "val_contrastive_loss": val_contrastive_loss,
            "val_ce_loss": val_ce_loss,
            "val_factuality_score": val_factuality,
            "val_toxicity": val_toxicity,
            "val_factuality_improvement": val_factuality - original_val_factuality,
            "val_toxicity_improvement": original_val_toxicity - val_toxicity,
        })
        
        # Report metrics together
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Contrastive Loss: {avg_train_contrastive_loss:.4f}, CE Loss: {avg_train_ce_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Contrastive Loss: {val_contrastive_loss:.4f}, CE Loss: {val_ce_loss:.4f}")
        logger.info(f"Val Factuality Score: {val_factuality:.4f}")
        logger.info(f"Val Toxicity Score: {val_toxicity:.4f}")
        logger.info(f"Val Factuality Improvement: {val_factuality - original_val_factuality:.4f}")
        logger.info(f"Val Toxicity Improvement: {original_val_toxicity - val_toxicity:.4f}")
        logger.info("---")
    
    # Final evaluation
    final_val_loss, final_val_contrastive_loss, final_val_ce_loss, final_val_factuality, final_val_toxicity = evaluate_w_toxicity(
        model, tokenizer, val_loader, device, factuality_detector, logger, criterion
    )
    logger.info("Final Validation Results:")
    logger.info(f"Loss: {final_val_loss:.4f}, Contrastive Loss: {final_val_contrastive_loss:.4f}, CE Loss: {final_val_ce_loss:.4f}")
    logger.info(f"Factuality Score: {final_val_factuality:.4f}")
    logger.info(f"Factuality Improvement: {final_val_factuality - original_val_factuality:.4f}")
    logger.info(f"Toxicity Score: {final_val_toxicity:.4f}")
    logger.info(f"Toxicity Improvement: {original_val_toxicity - final_val_toxicity:.4f}")

def evaluate_w_toxicity(model, tokenizer, val_loader, device, factuality_detector, logger, criterion):
    model.eval()
    total_val_loss = 0
    total_val_contrastive_loss = 0
    total_val_ce_loss = 0
    total_factuality = 0
    total_toxicity = 0
    num_samples = 0

    # Lists to store references, hypotheses, and source texts
    all_references = []
    all_hypotheses = []
    all_sources = []  # To store source texts

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Prepare data
            input_ids = batch['contrast_input_ids'].to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
            target_ids = batch['contrast_target_input_ids'].to(device)
            contrast_ne_masks = batch['contrast_ne_masks'].to(device)
            valid_contrast = batch['valid_contrast'].to(device)
            positive_contrast = batch['positive_contrast'].to(device)
            sample = {
                'contrast_ne': contrast_ne_masks,
                'valid_contrast': valid_contrast,
                'positive_contrast': positive_contrast
            }
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_ids,
                output_hidden_states=True,
                return_dict=True
            )
            
            lm_logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            loss, lm_loss, contrastive_loss = criterion(
                lm_logits=lm_logits,
                target_ids=target_ids,
                hidden_states=hidden_states,
                sample=sample
            )
            
            total_val_loss += loss.item()
            total_val_contrastive_loss += contrastive_loss.item()
            total_val_ce_loss += lm_loss.item()
            
            # Evaluate factuality and toxicity
            generated_ids = torch.argmax(lm_logits, dim=-1)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            target_texts = tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            source_ids = batch['src_input_ids']  # Get source input IDs
            source_texts = tokenizer.batch_decode(source_ids, skip_special_tokens=True)
            
            # Store references, hypotheses, and sources
            all_hypotheses.extend(generated_texts)
            all_references.extend(target_texts)
            all_sources.extend(source_texts)
            
            # Factuality scores
            factuality_scores = factuality_detector.evaluate(generated_texts)
            total_factuality += sum(factuality_scores)
            
            # Toxicity scores
            toxicity_scores = evaluate_toxicity(generated_texts)
            total_toxicity += sum(toxicity_scores)
            
            num_samples += len(generated_texts)
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_contrastive_loss = total_val_contrastive_loss / len(val_loader)
    avg_val_ce_loss = total_val_ce_loss / len(val_loader)
    avg_factuality = total_factuality / num_samples
    avg_toxicity = total_toxicity / num_samples

    # Compute BLEU score
    bleu = corpus_bleu(all_hypotheses, [all_references])
    bleu_score = bleu.score  # SacreBLEU returns an object with the score attribute

    # Compute ROUGE scores
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(all_references, all_hypotheses):
        scores = rouge_scorer_instance.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / num_samples
    avg_rouge2 = sum(rouge2_scores) / num_samples
    avg_rougeL = sum(rougeL_scores) / num_samples

    # Log BLEU and ROUGE scores
    logger.info(f"Validation BLEU Score: {bleu_score:.4f}")
    logger.info(f"Validation ROUGE-1 F1 Score: {avg_rouge1:.4f}")
    logger.info(f"Validation ROUGE-2 F1 Score: {avg_rouge2:.4f}")
    logger.info(f"Validation ROUGE-L F1 Score: {avg_rougeL:.4f}")

    # Also log to wandb
    wandb.log({
        "val_bleu": bleu_score,
        "val_rouge1": avg_rouge1,
        "val_rouge2": avg_rouge2,
        "val_rougeL": avg_rougeL,
    })

    # Print summaries for a few samples
    num_samples_to_print = 8  # Adjust the number of samples you want to print
    logger.info("Sample Summaries from Validation Set:")
    for i in range(num_samples_to_print):
        idx = random.randint(0, num_samples - 1)
        logger.info(f"Sample {i+1}:")
        logger.info(f"Source Text: {all_sources[idx]}")
        logger.info(f"Reference Summary: {all_references[idx]}")
        logger.info(f"Generated Summary: {all_hypotheses[idx]}")
        logger.info("---")
    
    table = wandb.Table(columns=["Source Text", "Reference Summary", "Generated Summary"])
    for i in range(num_samples_to_print):
        idx = random.randint(0, num_samples - 1)
        table.add_data(all_sources[idx], all_references[idx], all_hypotheses[idx])

    # Log the table to wandb
    wandb.log({"Sample Summaries": table})

    model.train()
    return avg_val_loss, avg_val_contrastive_loss, avg_val_ce_loss, avg_factuality, avg_toxicity


def main():
    logger.info("Starting main function...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_data', nargs='+', choices=['original', 'bt', 'all'], default=['original'], help='Positive data options')
    parser.add_argument('--neg_data', nargs='+', choices=['original', 'entity_swap', 'low_confidence', 'all'], default=['original'], help='Negative data options')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model to train')
    args = parser.parse_args()

    # Process 'all' option
    all_pos_options = ['original', 'bt']
    all_neg_options = ['original', 'entity_swap', 'low_confidence']
    
    if 'all' in args.pos_data:
        args.pos_data = all_pos_options
    if 'all' in args.neg_data:
        args.neg_data = all_neg_options

    # Ensure 'original' is always included
    if 'original' not in args.pos_data:
        args.pos_data.insert(0, 'original')
    if 'original' not in args.neg_data:
        args.neg_data.insert(0, 'original')

    # Define data paths based on arguments
    data_dir = '/gpfs/home/bsk18/factual-bias-mitigation/data/tldr'  # Update with your actual data directory
    pos_data_paths = []
    neg_data_paths = []

    for opt in args.pos_data:
        if opt == 'original':
            pos_data_paths.append(os.path.join(data_dir, 'pos'))
        else:
            pos_data_paths.append(os.path.join(data_dir, f'pos_{opt}'))
    for opt in args.neg_data:
        if opt == 'original':
            neg_data_paths.append(os.path.join(data_dir, 'neg'))
        else:
            neg_data_paths.append(os.path.join(data_dir, f'neg_{opt}'))

    # Combine data from all specified paths
    pos_data = combine_data([read_files(path, logger) for path in pos_data_paths])
    neg_data = combine_data([read_files(path, logger) for path in neg_data_paths])

    val_pos_data = read_files('/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/validation/pos', logger, splits=['validation'])
    val_neg_data = read_files('/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/validation/neg', logger, splits=['validation'])

    # Load index mappings
    index_dir = '/gpfs/home/bsk18/factual-bias-mitigation/data/tldr/indices/pos_all_neg_original'  # Update with the directory where index files are stored
    train_pos_indices = load_indices(os.path.join(index_dir, 'train.positive.index'))
    train_neg_indices = load_indices(os.path.join(index_dir, 'train.negative.index'))
    val_pos_indices = load_indices(os.path.join(index_dir, 'validation.positive.index'))
    val_neg_indices = load_indices(os.path.join(index_dir, 'validation.negative.index'))

    # Initialize the factuality detector
    factuality_detector = FactualityDetector("buseskorkmaz/factual-bias-mitigation-models")
    logger.info("Initialized FactualityDetector")

    model_name = args.model_name

    # Initialize the model
    if model_name == "gpt2":
        config = GPT2Config.from_pretrained('gpt2')
        config.output_hidden_states = True
        model = ContrastiveGPT2Model(config)
    else:
        # Load other models as needed
        pass
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    train_dataset = ContrastiveTranslationDataset(
        src_texts=pos_data['train']['source'],
        pos_texts=pos_data['train']['target'],
        neg_texts=neg_data['train']['target'],
        pos_indices=train_pos_indices,
        neg_indices=train_neg_indices,
        tokenizer=tokenizer,
        max_length=512,
        max_neg_samples=5,
        cl_seed=0
    )

    val_dataset = ContrastiveTranslationDataset(
        src_texts=val_pos_data['validation']['source'],
        pos_texts=val_pos_data['validation']['target'],
        neg_texts=val_neg_data['validation']['target'],
        pos_indices=val_pos_indices,
        neg_indices=val_neg_indices,
        tokenizer=tokenizer,
        max_length=512,
        max_neg_samples=5,
        cl_seed=0
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=contrastive_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, collate_fn=contrastive_collate_fn
    )
    
    logger.info("Created datasets")

    # Calculate original factuality scores
    logger.info("Calculating original factuality scores...")
    avg_pos_factuality, avg_neg_factuality = calculate_original_factuality(
        val_dataset, factuality_detector, dataset_name='validation'
    )
    logger.info(f"Original Validation Positive Factuality Score: {avg_pos_factuality:.4f}")
    logger.info(f"Original Validation Negative Factuality Score: {avg_neg_factuality:.4f}")
    original_val_factuality = (avg_neg_factuality * len(val_neg_data) + avg_pos_factuality * len(val_pos_data))/ (len(val_neg_data) + len(val_pos_data))

    avg_pos_toxicity, avg_neg_toxicity = calculate_original_toxicity(
        val_dataset, dataset_name='validation'
        )
    logger.info(f"Original Validation Positive Toxicity Score: {avg_pos_toxicity:.4f}")
    logger.info(f"Original Validation Negative Toxicity Score: {avg_neg_toxicity:.4f}")
    original_val_toxicity = (avg_neg_toxicity * len(val_neg_data) + avg_pos_toxicity * len(val_pos_data))/ (len(val_neg_data) + len(val_pos_data))

    logger.info(f"Original Validation Factuality Score: {original_val_factuality:.4f}")
    logger.info(f"Original Validation Toxicity Score: {original_val_toxicity:.4f}")

    # Create config dictionary
    config = {
        "contrastive_weight": 1.0,
        "tau": 1.0,
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 8,
        "pos_data_option": args.pos_data,
        "neg_data_option": args.neg_data,
        "original_val_factuality": original_val_factuality,
    }

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    wandb.init(
        project='gpt2_contrastive_tldr',
        name=f"{model_name}-{'_'.join(args.pos_data)}-{'_'.join(args.neg_data)}",
    )

    wandb.config.update(config)

    criterion = ContrastiveLoss(
        alpha=config["contrastive_weight"],
        tau=config["tau"],
        padding_idx=tokenizer.pad_token_id
    )

    train(
        model, 
        tokenizer, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        config,
        epochs=config["epochs"], 
        factuality_detector=factuality_detector, 
        original_val_factuality=original_val_factuality,
        original_val_toxicity=original_val_toxicity,
        criterion=criterion
    )

    # Save the model
    output_dir = f"outputs/{model_name}_{'_'.join(args.pos_data)}_{'_'.join(args.neg_data)}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model and tokenizer saved to {output_dir}")

    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
