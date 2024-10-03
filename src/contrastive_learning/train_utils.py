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
    batch_size = pos_embeddings.shape[0]
    embedding_dim = pos_embeddings.shape[1]

    # Adjust for neg_embeddings
    if neg_embeddings.dim() == 2:
        # Only one negative per sample
        num_negatives = 1
        neg_embeddings = neg_embeddings.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
    else:
        num_negatives = neg_embeddings.shape[1]
    
    print(f"pos_embeddings.shape: {pos_embeddings.shape}")
    print(f"neg_embeddings.shape: {neg_embeddings.shape}")
    print(f"batch_size: {batch_size}")
    print(f"num_negatives: {num_negatives}")
    print(f"embedding_dim: {embedding_dim}")
    
    # Expand pos_embeddings to match neg_embeddings
    pos_embeddings_expanded = pos_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
    
    # Compute similarity matrix
    similarity_matrix = torch.sum(pos_embeddings_expanded * neg_embeddings, dim=-1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
    contrastive_loss_value = torch.nn.functional.cross_entropy(similarity_matrix, labels)
    
    # Cross-entropy loss for language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    ce_loss_value = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    
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
def evaluate_w_toxicity(model, tokenizer, data_loader, device, factuality_detector, logger, temperature=0.5, max_samples=100):
    model.eval()
    total_loss = 0
    total_contrastive_loss = 0
    total_ce_loss = 0
    total_factuality_score = 0
    total_toxicity_score = 0
    num_samples = 0
    pos_samples = 0
    neg_samples = 0

    # Get the total number of samples and the ratio
    total_dataset_size = data_loader.dataset.pos_length + data_loader.dataset.neg_length
    pos_ratio = data_loader.dataset.pos_length / total_dataset_size
    neg_ratio = data_loader.dataset.neg_length / total_dataset_size

    # Calculate the target number of positive and negative samples
    target_pos_samples = int(max_samples * pos_ratio)
    target_neg_samples = max_samples - target_pos_samples

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if num_samples >= max_samples:
                break

            # Forward pass
            embeddings, logits = model(batch['combined_ids'].to(device), batch['combined_attention_mask'].to(device))

            # Generate outputs for factuality and toxicity scoring
            generated_ids = model.generate(batch['source_ids'].to(device), batch['source_attention_mask'].to(device))

            for i in range(embeddings.shape[0]):
                if num_samples >= max_samples:
                    break

                is_positive = batch['is_positive'][i].item()

                # Ensure we maintain the ratio of positive to negative samples
                if is_positive and pos_samples >= target_pos_samples:
                    continue
                if not is_positive and neg_samples >= target_neg_samples:
                    continue

                full_prompt = batch['full_prompt'][i]
                generated_summary = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                # Debug logging
                logger.info(f"Debug: Factuality Detector Input")
                logger.info(f"Full Prompt: {full_prompt}")
                logger.info(f"Generated Summary: {generated_summary}")
                logger.info(f"Preprocessed Full Prompt: {preprocess_text(full_prompt)}")
                logger.info(f"Preprocessed Generated Summary: {preprocess_text(generated_summary)}")
                logger.info("---")
                
                # Calculate factuality score one sample at a time
                _, _, factuality_score = factuality_detector.generate_score(
                    [preprocess_text(full_prompt)], 
                    [preprocess_text(generated_summary)], 
                    summac_style=True
                )
                total_factuality_score += factuality_score

                # Calculate toxicity score
                toxicity_score = evaluate_toxicity(generated_summary)
                total_toxicity_score += toxicity_score

                num_samples += 1
                if is_positive:
                    pos_samples += 1
                else:
                    neg_samples += 1

                # Log some examples
                if num_samples % 10 == 0:
                    logger.info(f"Sample {num_samples}:")
                    logger.info(f"Is Positive: {is_positive}")
                    logger.info(f"Prompt: {full_prompt}")
                    logger.info(f"Generated: {generated_summary}")
                    logger.info(f"Factuality score: {factuality_score}")
                    logger.info(f"Toxicity score: {toxicity_score}")
                    logger.info("---")

            # Calculate loss
            if 'is_positive' in batch and batch['is_positive'].any():
                pos_indices = batch['is_positive'].nonzero(as_tuple=True)[0]
                neg_indices = (~batch['is_positive']).nonzero(as_tuple=True)[0]
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    pos_embeddings = embeddings[pos_indices]
                    neg_embeddings = embeddings[neg_indices]
                    pos_logits = logits[pos_indices]
                    pos_target_ids = batch['target_ids'][pos_indices].to(device)
                    
                    loss, contrastive_loss, ce_loss = combined_loss(pos_embeddings, neg_embeddings, pos_logits, pos_target_ids)
                    
                    total_loss += loss.item()
                    total_contrastive_loss += contrastive_loss.item()
                    total_ce_loss += ce_loss.item()

    avg_loss = total_loss / len(data_loader) if total_loss > 0 else 0
    avg_contrastive_loss = total_contrastive_loss / len(data_loader) if total_contrastive_loss > 0 else 0
    avg_ce_loss = total_ce_loss / len(data_loader) if total_ce_loss > 0 else 0
    avg_factuality_score = total_factuality_score / num_samples
    avg_toxicity_score = total_toxicity_score / num_samples

    logger.info(f"Evaluated on {num_samples} samples ({pos_samples} positive, {neg_samples} negative)")
    logger.info(f"Original dataset ratio - Positive: {pos_ratio:.2f}, Negative: {neg_ratio:.2f}")
    logger.info(f"Evaluated sample ratio - Positive: {pos_samples/num_samples:.2f}, Negative: {neg_samples/num_samples:.2f}")

    return avg_loss, avg_contrastive_loss, avg_ce_loss, avg_factuality_score, avg_toxicity_score

def contrastive_loss(pos_embeddings, neg_embeddings, temperature=1):
    batch_size, num_negatives = neg_embeddings.shape
    pos_embeddings = pos_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
    
    # Ensure both tensors have the same shape
    assert pos_embeddings.shape == neg_embeddings.shape, f"Shape mismatch: pos_embeddings {pos_embeddings.shape}, neg_embeddings {neg_embeddings.shape}"
    
    similarity_matrix = torch.sum(pos_embeddings * neg_embeddings, dim=-1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
    
    return torch.nn.functional.cross_entropy(similarity_matrix, labels)
