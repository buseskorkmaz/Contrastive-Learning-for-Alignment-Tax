import torch
from transformers import GPT2LMHeadModel, GPT2Config

class INLPGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, model_name_or_path, projection_matrix):
        config = GPT2Config.from_pretrained(model_name_or_path)
        super().__init__(config)
        
        # Load the pretrained model weights
        self.load_state_dict(GPT2LMHeadModel.from_pretrained(model_name_or_path).state_dict())
        
        self.projection_matrix = torch.nn.Parameter(projection_matrix, requires_grad=False)
        
        # Register the forward hook
        self.transformer.register_forward_hook(self._inlp_hook)

    def _inlp_hook(self, module, input_, output):
        if hasattr(output, 'last_hidden_state'):
            x = output.last_hidden_state
        else:
            return output

        # Apply the projection matrix to each token representation
        x = torch.matmul(self.projection_matrix.to(x.device), x.unsqueeze(-1)).squeeze(-1)
        
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