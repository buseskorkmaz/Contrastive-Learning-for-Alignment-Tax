from transformers import AutoModelForCausalLM, GPT2Tokenizer, AutoTokenizer, LlamaTokenizer


class CDAModel:
    def __init__(self, base_model_name, mode):
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/{base_model_name}-cda-{mode}")
        if base_model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif base_model_name == 'llama2-7b':
            self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", legacy=False)
        elif base_model_name == 'phi2':
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')