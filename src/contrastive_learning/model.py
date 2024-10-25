import torch

class ContrastiveJobDescriptionModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.transformer = model
        self.tokenizer = tokenizer
        self.transformer.resize_token_embeddings(len(tokenizer))
        # self.projection = torch.nn.Linear(self.transformer.config.n_embd, 128)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # For CausalLMOutputWithCrossAttentions, we need to use hidden_states
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1)
        logits = outputs.logits
        
        # return self.projection(pooled_output), logits
        return pooled_output, logits

    def generate(self, input_ids, attention_mask, pad_token_id=None, max_length=128):
        return self.transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            num_return_sequences=1,
            no_repeat_ngram_size=1,
            pad_token_id=pad_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            output_hidden_states=True  # Add this line
        )

# from transformers import AutoModelForCausalLM
# import torch

# class SummarizationModel(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, input_ids, attention_mask, labels=None):
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
#         # outputs.loss is computed internally
#         return outputs
