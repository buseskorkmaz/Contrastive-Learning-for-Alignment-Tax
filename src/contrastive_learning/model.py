import torch

class ContrastiveJobDescriptionModel(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.transformer = model
        self.tokenizer = tokenizer
        self.transformer.resize_token_embeddings(len(tokenizer))
        self.projection = torch.nn.Linear(self.transformer.config.n_embd, 256)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # For CausalLMOutputWithCrossAttentions, we need to use hidden_states
        last_hidden_state = outputs.hidden_states[-1]
        logits = outputs.logits
        
        # Use the mean of the last hidden state across the sequence length
        pooled_output = last_hidden_state.mean(dim=1)
        
        return self.projection(pooled_output), logits

    def generate(self, input_ids, attention_mask, max_length=256):
        return self.transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            output_hidden_states=True  # Add this line
        )