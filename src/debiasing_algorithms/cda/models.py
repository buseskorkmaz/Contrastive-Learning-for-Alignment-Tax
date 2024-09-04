from transformers import AutoTokenizer, AutoModelForCausalLM


class CDAGPT2Model:
    def __init__(self, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/gpt2-cda-{mode}")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/gpt2-cda-{mode}")

class CDALlama2Model:
    def __init__(self, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/llama2-7b-cda-{mode}")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/llama2-7b-cda-{mode}")


class CDAPhi2Model:
    def __init__(self, mode):
        self.tokenizer = AutoTokenizer.from_pretrained(f"ao9000/phi2-cda-{mode}")
        self.model = AutoModelForCausalLM.from_pretrained(f"ao9000/phi2-cda-{mode}")
    