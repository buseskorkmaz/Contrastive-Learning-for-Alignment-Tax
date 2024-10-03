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

def combined_loss(pos_embeddings, neg_embeddings, logits, labels, contrastive_weight=1, ce_weight=1, temperature=0.7):
    # Combine positive and negative embeddings
    embeddings = torch.cat([pos_embeddings.unsqueeze(1), neg_embeddings], dim=1)  # [batch_size, num_negatives+1, embedding_dim]

    # Compute similarity matrix
    similarity_matrix = torch.bmm(embeddings, embeddings.transpose(1, 2)) / temperature  # [batch_size, num_negatives+1, num_negatives+1]

    # We need to create labels for contrastive loss
    contrastive_labels = torch.arange(embeddings.size(1), device=embeddings.device).long()
    contrastive_loss_value = torch.nn.functional.cross_entropy(similarity_matrix.view(-1, embeddings.size(1)), contrastive_labels.repeat(embeddings.size(0)))

    # Cross-entropy loss
    logits_reshaped = logits.view(-1, logits.size(-1))
    labels_reshaped = labels.view(-1)
    # After reshaping logits and labels
    print(f"logits_reshaped.min(): {logits_reshaped.min().item()}, logits_reshaped.max(): {logits_reshaped.max().item()}")
    print(f"labels_reshaped.min(): {labels_reshaped.min().item()}, labels_reshaped.max(): {labels_reshaped.max().item()}")

    if torch.isnan(logits_reshaped).any():
        print("NaN detected in logits_reshaped")
    if torch.isinf(logits_reshaped).any():
        print("Inf detected in logits_reshaped")
    if torch.isnan(labels_reshaped).any():
        print("NaN detected in labels_reshaped")
    if torch.isinf(labels_reshaped).any():
        print("Inf detected in labels_reshaped")

    # Ensure labels are of type torch.LongTensor
    labels_reshaped = labels_reshaped.long()
    print(f"Labels dtype: {labels_reshaped.dtype}")
    invalid_labels = labels_reshaped[(labels_reshaped != -100) & ((labels_reshaped < 0) | (labels_reshaped >= logits.size(-1)))]
    if invalid_labels.numel() > 0:
        print(f"Found invalid labels: {invalid_labels}")

    ce_loss_value = torch.nn.functional.cross_entropy(
        logits_reshaped,
        labels_reshaped,
        ignore_index=-100
    )

    # Combined loss
    loss = contrastive_weight * contrastive_loss_value + ce_weight * ce_loss_value
    return loss, contrastive_loss_value, ce_loss_value

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

            is_positive = batch['is_positive'].item()  # Assuming batch size is 1

            # Ensure we maintain the ratio of positive to negative samples
            if is_positive and pos_samples >= target_pos_samples:
                continue
            if not is_positive and neg_samples >= target_neg_samples:
                continue

            input_ids = batch['combined_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            full_prompt = batch['full_prompt'][0]

            # Forward pass
            embeddings, logits = model(input_ids, attention_mask)
            if torch.isnan(logits).any():
                print("NaN detected in logits")
            if torch.isinf(logits).any():
                print("Inf detected in logits")

            # Generate summaries
            generated_ids = model.generate(
                input_ids=batch['pos_source_ids'].to(device),
                attention_mask=batch['pos_source_attention_mask'].to(device),
                max_length=128,
                pad_token_id=tokenizer.eos_token_id  # Ensure the pad_token_id is set
            )

            generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            if is_positive:
                # Handle labels for decoding
                labels_clean = labels.clone()
                labels_clean[labels_clean == -100] = tokenizer.pad_token_id

                labels_list = labels_clean.tolist()
                filtered_labels = [
                    [token_id for token_id in seq if token_id != tokenizer.pad_token_id]
                    for seq in labels_list
                ]

                reference_summary = tokenizer.decode(filtered_labels[0], skip_special_tokens=True)

                # Compute loss
                logits_reshaped = logits.view(-1, logits.size(-1))
                labels_reshaped = labels.view(-1)
                ce_loss_value = torch.nn.functional.cross_entropy(
                    logits_reshaped,
                    labels_reshaped,
                    ignore_index=-100
                )
                total_ce_loss += ce_loss_value.item()
                total_loss += ce_loss_value.item()

                pos_samples += 1

                # Log some examples
                if pos_samples % 10 == 0:
                    logger.info(f"Positive Sample {pos_samples}:")
                    logger.info(f"Full Prompt: {full_prompt}")
                    logger.info(f"Reference Summary: {reference_summary}")
                    logger.info(f"Generated Summary: {generated_summary}")
                    logger.info(f"Loss: {ce_loss_value.item()}")
                    logger.info("---")
            else:
                neg_samples += 1
                # Optionally, log negative samples every so often
                if neg_samples % 10 == 0:
                    logger.info(f"Negative Sample {neg_samples}:")
                    logger.info(f"Full Prompt: {full_prompt}")
                    logger.info(f"Generated Summary: {generated_summary}")
                    logger.info("---")

            # Calculate factuality and toxicity scores for both positive and negative samples
            _, _, factuality_score = factuality_detector.generate_score(
                [preprocess_text(full_prompt)],
                [preprocess_text(generated_summary)],
                summac_style=True
            )
            total_factuality_score += factuality_score

            toxicity_score = evaluate_toxicity(generated_summary)
            total_toxicity_score += toxicity_score

            num_samples += 1

            # Break if we've reached the desired number of samples
            if pos_samples >= target_pos_samples and neg_samples >= target_neg_samples:
                break

        # Calculate average losses and scores
        avg_loss = total_ce_loss / pos_samples if pos_samples > 0 else 0
        avg_factuality_score = total_factuality_score / num_samples if num_samples > 0 else 0
        avg_toxicity_score = total_toxicity_score / num_samples if num_samples > 0 else 0

        logger.info(f"Evaluated on {num_samples} samples ({pos_samples} positive, {neg_samples} negative)")
        logger.info(f"Original dataset ratio - Positive: {pos_ratio:.2f}, Negative: {neg_ratio:.2f}")
        logger.info(f"Evaluated sample ratio - Positive: {pos_samples/num_samples:.2f}, Negative: {neg_samples/num_samples:.2f}")

        return avg_loss, 0, avg_loss, avg_factuality_score, avg_toxicity_score

def contrastive_loss(pos_embeddings, neg_embeddings, temperature=1):
    batch_size, num_negatives = neg_embeddings.shape
    pos_embeddings = pos_embeddings.unsqueeze(1).expand(-1, num_negatives, -1)
    
    # Ensure both tensors have the same shape
    assert pos_embeddings.shape == neg_embeddings.shape, f"Shape mismatch: pos_embeddings {pos_embeddings.shape}, neg_embeddings {neg_embeddings.shape}"
    
    similarity_matrix = torch.sum(pos_embeddings * neg_embeddings, dim=-1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=pos_embeddings.device)
    
    return torch.nn.functional.cross_entropy(similarity_matrix, labels)
