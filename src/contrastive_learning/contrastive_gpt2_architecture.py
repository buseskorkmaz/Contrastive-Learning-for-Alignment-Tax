import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

class ContrastiveGPT2Model(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Optional classification head
        if hasattr(config, 'num_labels') and config.num_labels > 0:
            self.classification_head = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classification_head = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        src_select_index=None,
    ):
        # Get outputs from the base GPT2LMHeadModel
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Need hidden states for contrastive tasks
            return_dict=True,
        )

        # Get the hidden states and logits
        hidden_states = outputs.hidden_states  # Tuple of (layer1, layer2, ..., layerN)
        logits = outputs.logits  # Language modeling logits

        # Reorder outputs if src_select_index is provided
        if src_select_index is not None:
            logits = logits.index_select(0, src_select_index)
            hidden_states = tuple(h.index_select(0, src_select_index) for h in hidden_states)

        # Compute classification logits if classification head is defined
        classification_logits = None
        if self.classification_head is not None:
            # Use the hidden state of the last token (or apply pooling)
            cls_hidden_state = hidden_states[-1][:, -1, :]  # (batch_size, hidden_size)
            classification_logits = self.classification_head(cls_hidden_state)

        # Prepare the output
        if not return_dict:
            output = (logits,) + outputs[1:]
            if classification_logits is not None:
                output = output + (classification_logits,)
            return output

        # Return a dictionary including classification logits
        output = {
            'logits': logits,
            'past_key_values': outputs.past_key_values,
            'hidden_states': hidden_states,
            'attentions': outputs.attentions,
            'classification_logits': classification_logits,
        }
        if labels is not None:
            output['loss'] = outputs.loss
        return output
