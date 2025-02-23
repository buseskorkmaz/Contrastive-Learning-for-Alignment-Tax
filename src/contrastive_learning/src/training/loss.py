# src/training/loss.py - Custom loss functions for training
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch.distributed as dist

def label_smoothed_nll_loss(
    accelerator,
    lprobs: torch.Tensor, 
    target: torch.Tensor, 
    epsilon: float, 
    ignore_index: Optional[int] = None, 
    reduce: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fixed loss calculation with proper dimension handling"""
    # Handle empty batch case
    if lprobs.size(0) == 0 or target.size(0) == 0:
        if reduce:
            return torch.tensor(0.0, device=lprobs.device), torch.tensor(0.0, device=lprobs.device)
        else:
            return torch.zeros_like(target, dtype=torch.float), torch.zeros_like(target, dtype=torch.float)
    
    # Ensure target has right shape
    valid_positions = (target != -100).sum()
    rank = dist.get_rank()
    print(f"Rank {rank} valid positions: {valid_positions}", flush=True)
    accelerator.print(f"Rank {accelerator.process_index} valid positions: {valid_positions}")
    if valid_positions == 0:
        raise ValueError("Target tensor has no valid positions")

    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
        
    # Double check dimensions
    assert lprobs.size(0) == target.size(0), f"Batch size mismatch: {lprobs.size(0)} vs {target.size(0)}"
    
    # Get NLL loss
    nll_loss = -lprobs.gather(dim=-1, index=target)
    
    # Get smoothed loss against uniform distribution
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    
    # Handle padding if specified
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    
    # Reduce if requested
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    
    # Combine losses
    eps_i = epsilon / lprobs.size(-1)  # Uniform prior
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    
    return loss, nll_loss

@dataclass
class ContrastiveOutput:
    loss: torch.Tensor
    contrast_loss: Optional[torch.Tensor] = None
    ce_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class ContrastiveLoss(nn.Module):
    def __init__(self, accelerator, temperature=1.0, alpha=1.0, label_smoothing=0.0, ignore_prefix_size=0):
        super().__init__()
        self.accelerator = accelerator
        self.temperature = temperature
        self.alpha = alpha
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.padding_idx = None

    def forward(
        self,
        lm_logits: torch.Tensor,
        contrast_logits: torch.Tensor,
        model,
        sample: dict,
        logger,
        reduce: bool = True
    ) -> ContrastiveOutput:
        """
        Computes the CE loss from lm_logits + the contrastive loss from contrast_logits.
        """
        # Log initial stats
        logger.info(f"\nLoss computation debug:")
        logger.info(f"LM logits shape: {lm_logits.shape}")
        logger.info(f"Contrast logits shape: {contrast_logits.shape}")
        logger.info(f"Batch size: {lm_logits.shape[0]}")
        
        # Compute CE loss with dynamic scaling
        raw_ce_loss, _ = self.compute_loss(model, lm_logits, sample, logger, reduce=reduce)
        ce_loss = raw_ce_loss / sample["ntokens"]
        
        # Compute contrast loss
        contrast_loss = self.compute_contrast_loss(model, contrast_logits, sample, logger)
        
        # Dynamic scaling based on NE density and positive pairs
        ne_ratio = (sample["contrast_ne"] > 0).float().mean()
        pos_ratio = sample["positive_contrast"].float().sum() / sample["positive_contrast"].numel()

        # Determine if any of the negatives are toxic.
        toxic_flag = False
        # Here we assume that sample contains "augmentation_types" for each negative.
        # This is a simple check; you might need to refine it based on your data.
        if "augmentation_types" in sample:
            for aug in sample["augmentation_types"]:
                if aug == "toxic":
                    toxic_flag = True
                    break
        # If toxic negatives are present, increase their influence.
        toxic_weight = 1.5 if toxic_flag else 1.0

        contrast_loss = contrast_loss * toxic_weight

        # Scale losses based on content
        # ce_scale = 0.1      # Base CE scale
        # contrast_scale = min(0.9, 0.9 * (ne_ratio * 50))  # Scale contrast loss based on NE presence
        ce_scale = 1
        contrast_scale = 1
        
        # Apply scaling
        ce_loss = ce_loss * ce_scale
        contrast_loss = contrast_loss * contrast_scale
        
        # Combine losses
        final_loss = ce_loss + self.alpha * contrast_loss
        
        # Log detailed stats
        logger.info(f"Number of positive contrasts: {sample['positive_contrast'].sum().item()}")
        logger.info(f"Number of valid contrasts: {sample['valid_contrast'].sum().item()}")
        logger.info(f"NE density: {ne_ratio:.4f}")
        logger.info(f"Positive pair ratio: {pos_ratio:.4f}")
        logger.info(f"CE scale: {ce_scale:.4f}")
        logger.info(f"Contrast scale: {contrast_scale:.4f}")
        
        # Log raw and scaled losses
        logger.info(f"Raw CE loss: {(raw_ce_loss / sample['ntokens']).item():.4f}")
        logger.info(f"Scaled CE loss: {ce_loss.item():.4f}")
        logger.info(f"Raw contrast loss: {contrast_loss.item()/contrast_scale:.4f}")
        logger.info(f"Scaled contrast loss: {contrast_loss.item():.4f}")
        logger.info(f"Final loss: {final_loss.item():.4f}")
        
        # Compute representation stats
        with torch.no_grad():
            rep_norm = contrast_logits.norm(dim=-1)
            logger.info(f"Representation norms - mean: {rep_norm.mean():.3f}, min: {rep_norm.min():.3f}, max: {rep_norm.max():.3f}")
        
        return ContrastiveOutput(
            loss=final_loss,
            contrast_loss=contrast_loss,
            ce_loss=ce_loss,
            logits=None
        )
    
    def compute_contrast_loss(self, model, contrast_net_output, sample, logger):
        # First normalize input embeddings
        # contrast_net_output = F.normalize(contrast_net_output, dim=-1)
        logger.info(f"Using temperature: {self.temperature}")

        
        # Debug representation stats
        with torch.no_grad():
            logger.info(f"Contrast logits stats: min={contrast_net_output.min():.3f}, max={contrast_net_output.max():.3f}")
            
            # Check NE vs non-NE stats
            ne_mask = (sample["contrast_ne"] > 0)
            tokens_per_sample = ne_mask.sum(dim=1)
            logger.info(f"NE tokens per sample: {tokens_per_sample.tolist()}")
            
            ne_reps = contrast_net_output[ne_mask]
            non_ne_reps = contrast_net_output[~ne_mask]
            
            if ne_reps.numel() > 0:
                logger.info(f"NE representation norm: {ne_reps.norm(dim=-1).mean():.3f}")
            if non_ne_reps.numel() > 0:
                logger.info(f"Non-NE representation norm: {non_ne_reps.norm(dim=-1).mean():.3f}")
            
            active_ratio = ne_mask.float().mean()
            logger.info(f"Active NE tokens: {active_ratio:.2%}")

        # Get NE representations
        if sample["contrast_ne"] is not None:
            # Changed from == 0 to > 0 to properly mask non-NE tokens
            ne_mask = (sample["contrast_ne"] > 0).unsqueeze(-1)  # => [B, T, 1]
            
            # Zero out non-NE tokens
            ne_representation = contrast_net_output * ne_mask.float()
            
            # Sum NE tokens
            representation = ne_representation.sum(dim=1)  # => [B, projection_size]
            
            # Normalize by NE count with safe division
            ne_count = ne_mask.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1e-8)
            representation = representation / ne_count
        else:
            representation = contrast_net_output[:, -1, :]

        # Normalize pooled representations
        representation = F.normalize(representation, dim=-1)

    #     # Compute similarities with temperature scaling
    #     similarity = torch.matmul(representation, representation.transpose(0, 1))
    #     logger.info(f"Raw similarities before temp scaling: mean={similarity.mean():.3f}, min={similarity.min():.3f}, max={similarity.max():.3f}")
    #     similarity = similarity / self.temperature
        
    #     # Use log_softmax for better numerical stability
    #     log_similarity = F.log_softmax(similarity, dim=-1)
    #     logger.info(f"Log similarities after softmax: mean={log_similarity.mean():.3f}, min={log_similarity.min():.3f}, max={log_similarity.max():.3f}")

    #     # Mask invalid contrasts
    #     log_similarity = log_similarity.masked_fill(~sample["valid_contrast"], -1e9)
    #     logger.info(f"Log similarities after filterout: mean={log_similarity.mean():.3f}, min={log_similarity.min():.3f}, max={log_similarity.max():.3f}")
    #     positive_sims = similarity[sample["positive_contrast"]]
    #     logger.info(f"Positive pair similarities: mean={positive_sims.mean():.3f}, min={positive_sims.min():.3f}, max={positive_sims.max():.3f}")

    #     # Compute loss on positive pairs
    #     pos_loss = -log_similarity[sample["positive_contrast"]]
        
    #     # Safe averaging
    #     pos_count = sample["positive_contrast"].sum().clamp(min=1)
    #     loss = pos_loss.sum() / pos_count
        
    #     # negative_mask = ~sample["positive_contrast"] & sample["valid_contrast"]
    #     # if negative_mask.any():
    #     #     negative_sims = similarity[negative_mask]
    #     #     # Add repulsion loss
    #     #     repulsion_loss = F.relu(negative_sims - 0.5).mean()  # Push apart if similarity > 0.5
    #     #     loss = loss + 0.3 * repulsion_loss  # Small weight for stability
    #    # Get hardest negatives (highest similarity among negatives)
    #     negative_mask = ~sample["positive_contrast"] & sample["valid_contrast"]
    #     positive_mask = sample["positive_contrast"]
        
    #     # Get indices of positive pairs
    #     pos_row_idx, pos_col_idx = torch.where(positive_mask)
        
    #     hardest_negative_sims = []
    #     for pos_row, pos_col in zip(pos_row_idx, pos_col_idx):
    #         # Get negatives for this positive pair's row
    #         row_negatives = similarity[pos_row][negative_mask[pos_row]]
    #         if len(row_negatives) > 0:
    #             # Get hardest negative for this positive pair
    #             hardest_neg = row_negatives.max()
    #         else:
    #             # If no negatives, use a default value
    #             hardest_neg = torch.tensor(0.0, device=similarity.device)
    #         hardest_negative_sims.append(hardest_neg)
        
    #     hardest_negative_sims = torch.stack(hardest_negative_sims)
        
    #     # Get corresponding positive similarities
    #     positive_sims = similarity[pos_row_idx, pos_col_idx]
        
    #     # Now both positive_sims and hardest_negative_sims have the same size
    #     margin = 0.2
    #     hard_negative_loss = F.relu(margin - (positive_sims - hardest_negative_sims)).mean()
            
    #     loss = loss + 0.5 * hard_negative_loss

        # Use their similarity calculation approach
        similarity = torch.matmul(representation, representation.transpose(0, 1))
        similarity = similarity / self.temperature
        similarity = similarity.exp()  # Use exp directly like they do
        logger.info(f"EXp similarities after softmax: mean={similarity.mean():.3f}, min={similarity.min():.3f}, max={similarity.max():.3f}")

        
        # Their masking approach
        similarity = similarity.masked_fill(~sample["valid_contrast"], 0.)
        logger.info(f"EXp similarities after filtering: mean={similarity.mean():.3f}, min={similarity.min():.3f}, max={similarity.max():.3f}")

        
        # Their denominator handling
        denominator = similarity.sum(dim=-1, keepdim=True)
        similarity = similarity / torch.max(denominator, 1e-8 * torch.ones_like(denominator))
        
        # Get loss from positive pairs
        loss = similarity[sample["positive_contrast"]]
        loss = -loss.log()
        
        # Their safe averaging
        loss_denom = sample["positive_contrast"].sum()
        loss = loss.sum() / torch.max(loss_denom, 1e-8 * torch.ones_like(loss_denom))

        # Log loss components
        with torch.no_grad():
            logger.info(f"Average similarity: {similarity.mean():.3f}")
            # logger.info(f"Positive pair count: {pos_count.item()}")
            # logger.info(f"Hard negative loss: {hard_negative_loss.item():.3f}")
            logger.info(f"Loss value: {loss.item():.3f}")

        return loss
    
    def compute_loss(self, model, lm_logits, sample, logger, reduce=True):
        """Computes the label-smoothed CE loss (or standard CE if label_smoothing=0)."""
        # Convert to log-probs using model's helper if you prefer
        ce_pos = sample["ce_pos"]
        ce_lm_logits = lm_logits[ce_pos]
        ce_target = sample["target"][ce_pos]
        
        lprobs = model.get_normalized_probs(ce_lm_logits, log_probs=True)
        # target = sample["target"]

        # Reshape for CE
        lprobs = lprobs.view(-1, lprobs.size(-1))
        ce_target = ce_target.view(-1).to(lprobs.device)

        # Mask out -100 if necessary
        non_pad_mask = ce_target.ne(-100)
        lprobs = lprobs[non_pad_mask]
        ce_target = ce_target[non_pad_mask]

        # Call your label_smoothed_nll_loss helper
        loss, nll_loss = label_smoothed_nll_loss(
            self.accelerator,
            lprobs,
            ce_target,
            self.eps,
            ignore_index=-100,
            reduce=reduce
        )
        return loss, nll_loss

