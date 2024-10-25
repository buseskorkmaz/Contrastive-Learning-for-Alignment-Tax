import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, alpha=1.0, tau=1.0, padding_idx=0):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha  # Weight for the contrastive loss
        self.tau = tau      # Temperature parameter for contrastive loss
        self.padding_idx = padding_idx  # Padding token ID

        # Cross-entropy loss for language modeling
        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, lm_logits, target_ids, hidden_states, sample):
        """
        lm_logits: (batch_size, seq_len, vocab_size)
        target_ids: (batch_size, seq_len)
        hidden_states: (batch_size, seq_len, hidden_size)
        sample: dictionary containing necessary tensors for contrastive loss
        """
        # Language modeling loss
        lm_loss = self.ce_loss_fct(
            lm_logits.view(-1, lm_logits.size(-1)),  # Flatten to (batch_size * seq_len, vocab_size)
            target_ids.view(-1)                      # Flatten to (batch_size * seq_len)
        )

        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(hidden_states, sample)

        # Total loss
        loss = lm_loss + self.alpha * contrastive_loss

        return loss, lm_loss, contrastive_loss

    def compute_contrastive_loss(self, hidden_states, sample):
        """
        Computes the contrastive loss as per your original code.

        hidden_states: (batch_size, seq_len, hidden_size)
        sample: dictionary containing 'contrast_ne', 'valid_contrast', 'positive_contrast'
        """
        contrast_ne = sample.get("contrast_ne", None)  # (batch_size, seq_len)
        batch_size = hidden_states.size(0)

        if contrast_ne is not None:
            # Masked fill positions where contrast_ne == 0
            ne_mask = (contrast_ne == 0).unsqueeze(-1)  # (batch_size, seq_len, 1)
            ne_representation = hidden_states.masked_fill(ne_mask, 0)  # B x T x C

            # Sum over time dimension
            representation = ne_representation.sum(dim=1)  # B x C

            # Compute denominator (number of non-zero positions per sample)
            representation_ne_denom = contrast_ne.sum(dim=1, keepdim=True)  # B x 1

            # Avoid division by zero
            representation_ne_denom = representation_ne_denom.masked_fill(representation_ne_denom == 0, 1e-8)

            # Normalize representation
            representation = representation / representation_ne_denom
        else:
            # Use last token's hidden state
            representation = hidden_states[:, -1, :]  # B x C

        # Compute norms
        representation_n = representation.norm(dim=-1, keepdim=True)  # B x 1

        # Avoid division by zero
        representation_n = representation_n.masked_fill(representation_n == 0, 1e-8)

        # Normalize representations
        representation_norm = representation / representation_n

        # Compute similarity matrix
        similarity = torch.matmul(representation_norm, representation_norm.transpose(0, 1))  # B x B

        # Divide by temperature
        similarity = similarity / self.tau

        # Exponentiate similarities
        similarity = torch.exp(similarity)

        # Mask invalid contrasts
        valid_contrast = sample["valid_contrast"]  # B x B
        similarity = similarity.masked_fill(~valid_contrast, 0.0)

        # Compute denominator
        denominator = similarity.sum(dim=-1, keepdim=True)  # B x 1

        # Avoid division by zero
        denominator = denominator.masked_fill(denominator == 0, 1e-8)

        # Compute denom_similarity
        denom_similarity = similarity / denominator  # B x B

        # Select positive contrasts
        positive_contrast = sample["positive_contrast"]  # B x B
        loss_values = denom_similarity[positive_contrast]  # N_positives

        # Compute negative log likelihood
        loss = -torch.log(loss_values)

        # Sum losses
        loss_denom = positive_contrast.sum()  # Scalar

        # Avoid division by zero
        if loss_denom == 0:
            loss_denom = torch.tensor(1e-8, device=loss.device)

        loss = loss.sum() / loss_denom

        return loss
