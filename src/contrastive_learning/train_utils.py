import torch
from tqdm import tqdm
import re
from src.contrastive_learning.utils import evaluate_toxicity

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string inputs
    pattern = r"Based on the original.*$"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip().lower()

def combined_loss(pos_embeddings, neg_embeddings, logits, targets, contrastive_weight=1, ce_weight=16, temperature=0.7):
    # Contrastive loss
    batch_size, num_negatives = neg_embeddings.shape
    pos_embeddings = pos_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
    similarity_matrix = torch.sum(pos_embeddings * neg_embeddings, dim=-1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
    contrastive_loss_value = torch.nn.functional.cross_entropy(similarity_matrix, labels)

    # Cross-entropy loss for language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    ce_loss_value = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Combined loss
    loss = contrastive_weight * contrastive_loss_value + ce_weight * ce_loss_value
    return loss, contrastive_weight * contrastive_loss_value, ce_weight * ce_loss_value

def evaluate_w_fairness(model, tokenizer, data_loader, device, diversity_evaluator, factuality_detector, logger, temperature=0.5):
    model.eval()
    total_loss = 0
    total_contrastive_loss = 0
    total_ce_loss = 0
    total_factuality_score = 0
    total_fairness_score = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Forward pass
            pos_embeddings, pos_logits = model(batch['pos_combined_ids'].to(device), batch['pos_combined_attention_mask'].to(device))
            neg_embeddings, _ = model(batch['neg_combined_ids'].to(device), batch['neg_combined_attention_mask'].to(device))

            # Calculate individual losses
            loss, contrastive_loss, ce_loss = combined_loss(pos_embeddings, neg_embeddings, pos_logits, batch['pos_target_ids'].to(device))
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_ce_loss += ce_loss.item()

            # Generate outputs for factuality scoring
            generated_ids = model.generate(batch['pos_source_ids'].to(device), batch['pos_source_attention_mask'].to(device))
            for i in range(pos_embeddings.shape[0]):
                full_prompt = batch['full_prompt'][i]
                generated_job_desc = tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                # Calculate factuality score - check if preprocess text calculate things correctly for multi-batch data
                _, _, factuality_score = factuality_detector.generate_score([preprocess_text(full_prompt)], [preprocess_text(generated_job_desc)], summac_style=True)
                total_factuality_score += factuality_score
                
                if num_samples % 10 == 0:
                    fairness_score = diversity_evaluator.calc_q_value(preprocess_text(generated_job_desc))
                    total_fairness_score += fairness_score

                # Log some examples
                if num_samples % 1 == 0:
                    logger.info(f"Sample {num_samples}:")
                    logger.info(f"Prompt: {full_prompt}")
                    logger.info(f"Generated: {generated_job_desc}")
                    logger.info(f"Factuality score: {factuality_score}")
                if num_samples % 10 == 0:    
                    logger.info(f"Fairness score: {fairness_score}")    
                logger.info("---")

            num_samples += pos_embeddings.shape[0]

    avg_loss = total_loss / len(data_loader)
    avg_contrastive_loss = total_contrastive_loss / len(data_loader)
    avg_ce_loss = total_ce_loss / len(data_loader)
    avg_factuality_score = total_factuality_score / num_samples
    avg_fairness_score = total_fairness_score/ (num_samples // 10)

    return avg_loss, avg_contrastive_loss, avg_ce_loss, avg_factuality_score, avg_fairness_score

def evaluate_w_toxicity(model, tokenizer, data_loader, device, factuality_detector, logger, temperature=0.5):
    model.eval()
    total_loss = 0
    total_contrastive_loss = 0
    total_ce_loss = 0
    total_factuality_score = 0
    total_toxicity_score = 0
    num_samples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Forward pass
            pos_embeddings, pos_logits = model(batch['pos_combined_ids'].to(device), batch['pos_combined_attention_mask'].to(device))
            neg_embeddings, neg_logits = model(batch['neg_combined_ids'].to(device), batch['neg_combined_attention_mask'].to(device))

            # Calculate individual losses
            loss, contrastive_loss, ce_loss = combined_loss(pos_embeddings, neg_embeddings, pos_logits, batch['pos_target_ids'].to(device))
            
            total_loss += loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_ce_loss += ce_loss.item()

            # Generate outputs for factuality and toxicity scoring
            pos_generated_ids = model.generate(batch['pos_source_ids'].to(device), batch['pos_source_attention_mask'].to(device))
            neg_generated_ids = model.generate(batch['neg_source_ids'].to(device), batch['neg_source_attention_mask'].to(device))

            all_prompts = []
            all_generated = []

            # Process positive samples
            for i in range(pos_embeddings.shape[0]):
                pos_full_prompt = batch['pos_full_prompt'][i]
                pos_generated_job_desc = tokenizer.decode(pos_generated_ids[i], skip_special_tokens=True)

                all_prompts.append(preprocess_text(pos_full_prompt))
                all_generated.append(preprocess_text(pos_generated_job_desc))

                # Calculate toxicity score for positive sample
                pos_toxicity_score = evaluate_toxicity(pos_generated_job_desc)
                total_toxicity_score += pos_toxicity_score

                num_samples += 1

            # Process negative samples
            for i in range(neg_embeddings.shape[0]):
                neg_full_prompt = batch['neg_full_prompt'][i]
                neg_generated_job_desc = tokenizer.decode(neg_generated_ids[i], skip_special_tokens=True)

                all_prompts.append(preprocess_text(neg_full_prompt))
                all_generated.append(preprocess_text(neg_generated_job_desc))

                # Calculate toxicity score for negative sample
                neg_toxicity_score = evaluate_toxicity(neg_generated_job_desc)
                total_toxicity_score += neg_toxicity_score

                num_samples += 1

            # Calculate factuality score for the batch
            _, _, batch_factuality_score = factuality_detector.generate_score(all_prompts, all_generated, summac_style=True)
            total_factuality_score += batch_factuality_score * len(all_prompts)

            # Log some examples
            if num_samples % 100 == 0:
                logger.info(f"Sample {num_samples}:")
                logger.info(f"Positive Prompt: {all_prompts[0]}")
                logger.info(f"Positive Generated: {all_generated[0]}")
                logger.info(f"Negative Prompt: {all_prompts[-1]}")
                logger.info(f"Negative Generated: {all_generated[-1]}")
                logger.info("---")

    avg_loss = total_loss / len(data_loader)
    avg_contrastive_loss = total_contrastive_loss / len(data_loader)
    avg_ce_loss = total_ce_loss / len(data_loader)
    avg_factuality_score = total_factuality_score / num_samples
    avg_toxicity_score = total_toxicity_score / num_samples

    return avg_loss, avg_contrastive_loss, avg_ce_loss, avg_factuality_score, avg_toxicity_score

def contrastive_loss(pos_embeddings, neg_embeddings, temperature=1):
    batch_size, num_negatives = neg_embeddings.shape
    pos_embeddings = pos_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
    
    # Ensure both tensors have the same shape
    assert pos_embeddings.shape == neg_embeddings.shape, f"Shape mismatch: pos_embeddings {pos_embeddings.shape}, neg_embeddings {neg_embeddings.shape}"
    
    similarity_matrix = torch.sum(pos_embeddings * neg_embeddings, dim=-1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
    
    return torch.nn.functional.cross_entropy(similarity_matrix, labels)
