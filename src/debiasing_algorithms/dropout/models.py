from transformers import AutoTokenizer, AutoModelForCausalLM


class CDAGPT2Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/gpt2-dropout")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/gpt2-dropout")

class CDALlama2Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/llama2-7b-dropout")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/llama2-7b-dropout")


class CDAPhi2Model:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/phi2-dropout")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/phi2-dropout")
    