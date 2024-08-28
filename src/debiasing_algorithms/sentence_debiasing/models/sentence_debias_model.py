import torch
from transformers import GPT2LMHeadModel, GPT2Config

class SentenceDebiasGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, model_name_or_path, bias_direction):
        config = GPT2Config.from_pretrained(model_name_or_path)
        super().__init__(config)
        
        # Load the pretrained model weights
        self.load_state_dict(GPT2LMHeadModel.from_pretrained(model_name_or_path).state_dict())
        
        self.bias_direction = torch.nn.Parameter(bias_direction, requires_grad=False)
        
        # Register the forward hook
        self.transformer.register_forward_hook(self._sentence_debias_hook)

    def _sentence_debias_hook(self, module, input_, output):
        if hasattr(output, 'last_hidden_state'):
            x = output.last_hidden_state
        else:
            return output

        # Ensure that everything is on the same device.
        bias_direction = self.bias_direction.to(x.device)
        
        # Get the shape of x
        shape = x.shape
        
        # Reshape x to (batch_size * seq_len, hidden_dim)
        x_reshaped = x.view(-1, shape[-1])
        
        # Reshape bias_direction to (1, hidden_dim)
        bias_direction = bias_direction.view(1, -1)
        
        # Compute the projection
        projection = torch.matmul(x_reshaped, bias_direction.t())
        
        # Reshape projection back to original shape
        projection = projection.view(*shape[:-1], 1)
        
        # Subtract the projection
        x = x - projection * bias_direction.view(*([1] * (len(shape) - 1)), -1)
        
        # Create a new output with the modified last_hidden_state
        new_output = output.__class__(
            last_hidden_state=x,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            cross_attentions=output.cross_attentions,
        )
        
        return new_output

    def generate(self, input_ids, attention_mask=None, max_length=None, num_return_sequences=1, output_scores=False, return_dict_in_generate=False, **kwargs):
        outputs = super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            **kwargs
        )
        
        return outputs