import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, tau=1.0):
        super(ContrastiveLoss, self).__init__()
        self.tau = tau  # Temperature parameter for contrastive loss

    def forward(self, hidden_states, sample):
        """
        hidden_states: (N, seq_len, hidden_size)
        sample: dictionary containing 'contrast_ne', 'valid_contrast', 'positive_contrast'
        """
        contrast_ne = sample.get("contrast_ne", None)  # (N, seq_len)
        if contrast_ne is not None:
            # Masked fill positions where contrast_ne == 0
            ne_mask = (contrast_ne == 0).unsqueeze(-1)  # (N, seq_len, 1)
            ne_representation = hidden_states.masked_fill(ne_mask, 0)  # N x T x C

            # Sum over time dimension
            representation = ne_representation.sum(dim=1)  # N x C

            # Compute denominator (number of non-zero positions per sample)
            representation_ne_denom = contrast_ne.sum(dim=1, keepdim=True)  # N x 1

            # Avoid division by zero
            eps = 1e-8
            representation_ne_denom = representation_ne_denom.masked_fill(representation_ne_denom == 0, eps)

            # Normalize representation
            representation = representation / representation_ne_denom
        else:
            # Use last token's hidden state
            representation = hidden_states[:, -1, :]  # N x C

        # Normalize representations
        representation_norm = F.normalize(representation, p=2, dim=1)  # N x C

        # Compute similarity matrix
        similarity = torch.matmul(representation_norm, representation_norm.transpose(0, 1))  # N x N

        # Divide by temperature
        similarity = similarity / self.tau

        # Mask invalid contrasts
        valid_contrast = sample["valid_contrast"].to(similarity.device)  # N x N

        # Replace invalid entries with a large negative value
        LARGE_NEGATIVE = -1e9
        similarity = similarity.masked_fill(~valid_contrast, LARGE_NEGATIVE)

        # **Add a small value to the diagonal to ensure at least one valid entry per row**
        similarity = similarity + torch.eye(similarity.size(0), device=similarity.device) * eps

        # For numerical stability, subtract the max value from each row
        max_sim, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - max_sim.detach()

        # Compute log probabilities
        exp_similarity = torch.exp(similarity)
        sum_exp = exp_similarity.sum(dim=1, keepdim=True)

        # Avoid division by zero in log
        sum_exp = sum_exp.masked_fill(sum_exp == 0, eps)

        log_probs = similarity - torch.log(sum_exp)

        # Check for non-finite values
        if not torch.isfinite(similarity).all():
            print("Non-finite values in similarity matrix")
        if not torch.isfinite(log_probs).all():
            print("Non-finite values in log probabilities")

        # Select positive contrasts
        positive_contrast = sample["positive_contrast"].to(log_probs.device)  # N x N
        loss_values = -log_probs[positive_contrast]  # K positive pairs

        # Handle case where loss_values might be empty
        if loss_values.numel() == 0:
            loss = torch.tensor(0.0, requires_grad=True).to(log_probs.device)
        else:
            loss = loss_values.mean()

        return loss
