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
    calculate_original_fairness,
)
from src.contrastive_learning.train_utils import(
    combined_loss,
    evaluate,
)
from src.contrastive_learning.model import ContrastiveJobDescriptionModel
from src.contrastive_learning.dataset import ContrastiveDataset
from src.factuality_detector import FactualityDetector
from src.debiasing_algorithms.sentence_debiasing.models.sentence_debias_model import SentenceDebiasGPT2LMHeadModel 
from src.debiasing_algorithms.inlp.models.inlp_model import INLPGPT2LMHeadModel
from src.debiasing_algorithms.autorefine.model import AutoRefine
from src.fairness_metrics.diversity_evaluator.evaluate import Diversity_Evaluator

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW 
from tqdm import tqdm
from transformers import AutoModel, GPT2Tokenizer
import random
random.seed(42)
import wandb
import argparse
# import nltk
# nltk.download('punkt')
# logging.disable(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(model, tokenizer, train_loader, val_loader, optimizer, device, epochs, diversity_evaluator, factuality_detector, original_val_factuality, original_val_fairness):
    logger.info(f"Original Validation Factuality Score: {original_val_factuality:.4f}")
    logger.info("Starting training...")
    for epoch in range(epochs):
        print_gpu_memory(epoch)
        model.train()
        total_train_loss = 0
        total_train_contrastive_loss = 0
        total_train_ce_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            pos_embeddings, pos_logits = model(batch['pos_combined_ids'].to(device), batch['pos_combined_attention_mask'].to(device))
            neg_embeddings, _ = model(batch['neg_combined_ids'].to(device), batch['neg_combined_attention_mask'].to(device))
            
            loss, contrastive_loss, ce_loss = combined_loss(pos_embeddings, neg_embeddings, pos_logits, batch['pos_target_ids'].to(device))
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_contrastive_loss += contrastive_loss.item()
            total_train_ce_loss += ce_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_contrastive_loss = total_train_contrastive_loss / len(train_loader)
        avg_train_ce_loss = total_train_ce_loss / len(train_loader)
        
        # Validation
        val_loss, val_contrastive_loss, val_ce_loss, val_factuality, val_fairness = evaluate(model, tokenizer, val_loader, device, diversity_evaluator, factuality_detector, logger)
        
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
            "val_fairness": val_fairness,
            "val_factuality_improvement": val_factuality - original_val_factuality,
            "val_fairness_improvement": val_fairness - original_val_fairness,
        })

        # Report metrics together
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Contrastive Loss: {avg_train_contrastive_loss:.4f}, CE Loss: {avg_train_ce_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Contrastive Loss: {val_contrastive_loss:.4f}, CE Loss: {val_ce_loss:.4f}")
        logger.info(f"Val Factuality Score: {val_factuality:.4f}")
        logger.info(f"Val Fairness Score: {val_fairness:.4f}")
        logger.info(f"Val Factuality Improvement: {val_factuality - original_val_factuality:.4f}")
        logger.info(f"Val Fairness Improvement: {val_fairness - original_val_fairness:.4f}")
        logger.info("---")

    # Final evaluation
    final_val_loss, final_val_contrastive_loss, final_val_ce_loss, final_val_factuality, final_val_fairness = evaluate(model, tokenizer, val_loader, device, diversity_evaluator, factuality_detector, logger)
    logger.info("Final Validation Results:")
    logger.info(f"Loss: {final_val_loss:.4f}, Contrastive Loss: {final_val_contrastive_loss:.4f}, CE Loss: {final_val_ce_loss:.4f}")
    logger.info(f"Factuality Score: {final_val_factuality:.4f}")
    logger.info(f"Factuality Improvement: {final_val_factuality - original_val_factuality:.4f}")
    logger.info(f"Fairness Score: {final_val_fairness:.4f}")
    logger.info(f"Fairness Improvement: {final_val_fairness - original_val_fairness:.4f}")

def main():
    logger.info("Starting main function...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_data', nargs='+', choices=['original', 'bt_fr', 'all'], default=['original'], help='Positive data options')
    parser.add_argument('--neg_data', nargs='+', choices=['original', 'entity_swap', 'low_confidence', 'all'], default=['original'], help='Negative data options')
    args = parser.parse_args()

    # Process 'all' option
    all_pos_options = ['original', 'bt_fr']
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
    pos_data_paths = ['/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/pos']
    neg_data_paths = ['/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/neg']

    for opt in args.pos_data:
        if opt != 'original':
            pos_data_paths.append(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/pos_{opt}')
    for opt in args.neg_data:
        if opt != 'original':
            neg_data_paths.append(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/neg_processed/{opt}')

    # Combine data from all specified paths
    pos_data = combine_data([read_files(path, logger) for path in pos_data_paths])
    neg_data = combine_data([read_files(path, logger) for path in neg_data_paths])

    diversity_evaluator = Diversity_Evaluator(logger)
    # diversity_evaluator = None
    factuality_detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")
    logger.info("Initialized Diversity_Evaluator and FactualityDetector")

    logger.info("Calculating original factuality scores...")
    original_val_factuality = calculate_original_factuality(pos_data['validation'], factuality_detector)
    original_test_factuality = calculate_original_factuality(pos_data['test'], factuality_detector)
    original_val_fairness = calculate_original_fairness(pos_data['validation'], fairness_evaluator=diversity_evaluator)
    original_test_fairness = calculate_original_fairness(pos_data['test'], fairness_evaluator=diversity_evaluator)
    logger.info(f"Original Validation Factuality Score: {original_val_factuality:.4f}")
    logger.info(f"Original Test Factuality Score: {original_test_factuality:.4f}")
    logger.info(f"Original Validation Fairness Score: {original_val_fairness:.4f}")
    logger.info(f"Original Test Fairness Score: {original_test_fairness:.4f}")

    model_name = "inlp-gpt2"

    if "sentence-debiasing" in model_name:
        mode = 'gender'
        bias_direction = torch.load(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/src/debiasing_algorithms/sentence_debiasing/subspaces/subspace_m-GPT2Model_c-gpt2_t-{mode}.pt')
        model = SentenceDebiasGPT2LMHeadModel('gpt2', bias_direction)
    elif "inlp" in model_name:
        mode = 'gender'
        projection_matrix = torch.load(f'/dccstor/autofair/busekorkmaz/factual-bias-mitigation/src/debiasing_algorithms/inlp/projection_matrix/projection_m-GPT2Model_c-gpt2_t-{mode}_s-0.pt')
        model = INLPGPT2LMHeadModel('gpt2', projection_matrix)
    elif "autorefine" in model_name:
        # not tested yet
        model = AutoRefine()
    else:
        model = AutoModel.from_pretrained(model_name)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    model = ContrastiveJobDescriptionModel(model, tokenizer)
    # model = ContrastiveJobDescriptionModel(model_name, tokenizer)
    logger.info(f"Initialized model: {model_name}")

    neg_samples_per_pos = 5  # You can adjust this number
    train_dataset = ContrastiveDataset(pos_data['train'], neg_data['train'], tokenizer, max_length=256, neg_samples_per_pos=neg_samples_per_pos)
    val_dataset = ContrastiveDataset(pos_data['validation'], neg_data['validation'], tokenizer, max_length=256, neg_samples_per_pos=neg_samples_per_pos)
    test_dataset = ContrastiveDataset(pos_data['test'], neg_data['test'], tokenizer, max_length=256, neg_samples_per_pos=neg_samples_per_pos)
    logger.info("Created datasets")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # batch_size is 1 because each item already contains multiple negatives
    val_loader = DataLoader(val_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)
    logger.info("Created DataLoaders")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)

    wandb.init(
        project='factual-bias-mitigation-scripts_train',
        name=f"inlp-{args.pos_data}-{args.neg_data}",
    )

    wandb.config.update({
        "model_name": model_name,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "epochs": 15,
        "neg_samples_per_pos": neg_samples_per_pos,
        "pos_data_option": args.pos_data,
        "neg_data_option": args.neg_data,
    })

    train(
        model, 
        tokenizer, 
        train_loader, 
        val_loader, 
        optimizer, 
        device, 
        epochs=15, 
        diversity_evaluator=diversity_evaluator, 
        factuality_detector=factuality_detector, 
        original_val_factuality=original_val_factuality,
        original_val_fairness=original_val_fairness,
    )

    # Create directory name
    dir_name = f"/dccstor/autofair/busekorkmaz/factual-bias-mitigation/outputs/inlp_{'_'.join(args.pos_data)}_{'_'.join(args.neg_data)}"
    os.makedirs(dir_name, exist_ok=True)

    logger.info(f"Saving fine-tuned model in directory: {dir_name}")
    
    # Save the model
    model_path = os.path.join(dir_name, "model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved as: {model_path}")

    # Create config dictionary
    config = {
        "ce_weight": 16,
        "model_name": model_name,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "epochs": 15,
        "neg_samples_per_pos": neg_samples_per_pos,
        "pos_data_option": args.pos_data,
        "neg_data_option": args.neg_data,
        "original_val_factuality": original_val_factuality,
        "original_test_factuality": original_test_factuality
    }

    # Save config.json
    config_path = os.path.join(dir_name, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Config saved as: {config_path}")

    logger.info("Evaluating on test set...")
    test_loss, test_contrastive_loss, test_ce_loss, test_factuality, test_fairness = evaluate(model, tokenizer, test_loader, device, diversity_evaluator, factuality_detector, logger)
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info(f"Final CE Loss: {test_ce_loss:.4f}")
    logger.info(f"Final CL Loss: {test_contrastive_loss:.4f}")
    logger.info(f"Final Test Factuality Score: {test_factuality:.4f}")
    logger.info(f"Final Test Fairness Score: {test_fairness:.4f}")
    logger.info(f"Test Factuality Improvement: {test_factuality - original_test_factuality:.4f}")
    logger.info(f"Test Fairness Improvement: {test_fairness - original_test_fairness:.4f}")

    # Log test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_contrastive_loss": test_contrastive_loss,
        "test_ce_loss": test_ce_loss,
        "test_factuality_score": test_factuality,
        "test_fairness_score": test_fairness,
        "test_factuality_improvement": test_factuality - original_test_factuality,
        "test_fairness_improvement": test_fairness - original_test_fairness,
    })

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()