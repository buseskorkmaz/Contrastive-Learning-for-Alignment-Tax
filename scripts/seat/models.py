from functools import partial

import torch
import transformers

from peft import AutoPeftModelForCausalLM
from debias.self_debias.modeling import GPT2Wrapper, Llama2Wrapper, Phi2Wrapper
from debias.self_debias.modeling import MaskedLMWrapper

################################ Modified ###############################################################
from transformers import (
    BitsAndBytesConfig,
)
from pathlib import Path
import yaml
# import bias_bench
# # Get the base project path from the bias_bench module
# BASE_PATH = Path(bias_bench.__path__[0]).parent
# # Construct the path to the config file
# CONFIG_FILE = BASE_PATH / 'config.yml'
# access_token = ''
# PERSPECTIVE_API_KEY = ''
# try:
#     with open(CONFIG_FILE) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
#     access_token = config['HF_KEY']
#     PERSPECTIVE_API_KEY = config['PERSPECTIVE_AI_KEY']
# except FileNotFoundError:
#     print('No config file found. API keys will not be loaded.')

# from huggingface_hub import login
# login(token=access_token)

class AutoModelForCausalLM:
    """"To load huggingface models"""
    def __new__(self, model_name_or_path):
        return transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                                 output_hidden_states=True)

class PhiForCausalLM:
    def __new__(self, model_name_or_path):
        return transformers.PhiForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                           output_hidden_states=True)

class LlamaForCausalLM:
    def __new__(self, model_name_or_path):
        return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                             output_hidden_states=True)

class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path, return_dict=True,
                                                             output_hidden_states=True)

# Quantized versions of the models
# class PhiForCausalLM_Quantized:
#     def __new__(self, model_name_or_path):
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_use_double_quant=True,
#         )
#         return transformers.PhiForCausalLM.from_pretrained(model_name_or_path,
#                                                            return_dict=True,
#                                                            output_hidden_states=True,
#                                                            quantization_config=bnb_config,
#                                                            device_map="auto")
# class LlamaForCausalLM_Quantized:
#     def __new__(self, model_name_or_path):
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#         )
#         return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path,
#                                                              return_dict=True,
#                                                              output_hidden_states=True,
#                                                              quantization_config=bnb_config,
#                                                              device_map="auto")

# Self debias models
class SelfDebiasLlama2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = Llama2Wrapper(model_name_or_path, use_cuda=True)
        return model

class SelfDebiasPhi2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = Phi2Wrapper(model_name_or_path, use_cuda=True)
        return model

class SelfDebiasGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = GPT2Wrapper(model_name_or_path, use_cuda=True)
        return model


# CDA Models
class CDAPhi2LMHeadModel:
    def __new__(self, model_name_or_path):
        # Method 1: Combine adaptors, unload model then load
        # model = transformers.PhiForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model


class CDALlama2LMHeadModel:
    def __new__(self, model_name_or_path):
        # Method 1: Combine adaptors, unload model then load
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model

class CDAGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        # model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        # return model

        # Method 1: Combine adaptors, unload model then load
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model

# Sentence Debias Models & INLP
class PhiForCausalLM_NonBFloat16:
    def __new__(self, model_name_or_path):
        return transformers.PhiForCausalLM.from_pretrained(model_name_or_path, output_hidden_states=True)

class LlamaForCausalLM_NonBFloat16:
    def __new__(self, model_name_or_path):
        return transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, output_hidden_states=True)

class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path, output_hidden_states=True)


# Sentence Debias Models
class _SentenceDebiasModel:
    def __init__(self, model_name_or_path, bias_direction):
        def _hook(module, input_, output, bias_direction):
            # Debias the last hidden state.
            # x = output["last_hidden_state"]
            # Modified
            if 'last_hidden_state' in output:
                x = output["last_hidden_state"]
            else:
                x = output['hidden_states'][-1]

            # Ensure that everything is on the same device.
            bias_direction = bias_direction.to(x.device)

            # Debias the representation.
            for t in range(x.size(1)):
                x[:, t] = x[:, t] - torch.ger(
                    torch.matmul(x[:, t], bias_direction), bias_direction
                ) / bias_direction.dot(bias_direction)

            # Update the output.
            # output["last_hidden_state"] = x
            if 'last_hidden_state' in output:
                output["last_hidden_state"] = x
            else:
                # output['hidden_states'][-1] = x
                temp = list(output['hidden_states'])
                temp[-1] = x # Update the last hidden state
                output['hidden_states'] = tuple(temp)

            return output

        self.func = partial(_hook, bias_direction=bias_direction)

class SentenceDebiasPhi2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.PhiForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                            output_hidden_states=True)
        model.model.register_forward_hook(self.func)
        return model

class SentenceDebiasLlama2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                            output_hidden_states=True)
        model.model.register_forward_hook(self.func)
        return model

class SentenceDebiasGPT2LMHeadModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path, return_dict=True,
                                                            output_hidden_states=True)
        model.transformer.register_forward_hook(self.func)
        return model

# INLP Models
class _INLPModel:
    def __init__(self, model_name_or_path, projection_matrix):
        def _hook(module, input_, output, projection_matrix):
            # Debias the last hidden state.
            # x = output["last_hidden_state"]
            if 'last_hidden_state' in output:
                x = output["last_hidden_state"]
            else:
                x = output['hidden_states'][-1]

            # Ensure that everything is on the same device.
            projection_matrix = projection_matrix.to(x.device)

            for t in range(x.size(1)):
                x[:, t] = torch.matmul(projection_matrix, x[:, t].T).T

            # Update the output.
            # output["last_hidden_state"] = x
            if 'last_hidden_state' in output:
                output["last_hidden_state"] = x
            else:
                # output['hidden_states'][-1] = x
                temp = list(output['hidden_states'])
                temp[-1] = x # Update the last hidden state
                output['hidden_states'] = tuple(temp)

            return output

        self.func = partial(_hook, projection_matrix=projection_matrix)

class INLPPhi2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.PhiForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                            output_hidden_states=True)
        model.model.register_forward_hook(self.func)
        return model

class INLPLlama2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path, return_dict=True,
                                                              output_hidden_states=True)
        model.model.register_forward_hook(self.func)
        return model

class INLPGPT2LMHeadModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path, return_dict=True,
                                                              output_hidden_states=True)
        model.transformer.register_forward_hook(self.func)
        return model

# Dropout models
class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        # model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        # return model

        # Method 1: Combine adaptors, unload model then load
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model

class DropoutPhi2LMHeadModel:
    def __new__(self, model_name_or_path):
        # model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        # return model

        # Method 1: Combine adaptors, unload model then load
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model

class DropoutLlama2LMHeadModel:
    def __new__(self, model_name_or_path):
        # model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        # return model

        # Method 1: Combine adaptors, unload model then load
        # model = transformers.LlamaForCausalLM.from_pretrained(model_name_or_path)

        # Method 2: Load model directly using peft
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True,
            quantization_config=bnb_config,
        )
        return model

############################################################################################################


class BertModel:
    def __new__(self, model_name_or_path):
        return transformers.BertModel.from_pretrained(model_name_or_path)


class AlbertModel:
    def __new__(self, model_name_or_path):
        return transformers.AlbertModel.from_pretrained(model_name_or_path)


class RobertaModel:
    def __new__(self, model_name_or_path):
        return transformers.RobertaModel.from_pretrained(model_name_or_path)


class BertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.BertForMaskedLM.from_pretrained(model_name_or_path)


class AlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)


class RobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        return transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)


class SentenceDebiasBertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaModel(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


# class SentenceDebiasGPT2Model(_SentenceDebiasModel):
#     def __new__(self, model_name_or_path, bias_direction):
#         super().__init__(self, model_name_or_path, bias_direction)
#         model = transformers.GPT2Model.from_pretrained(model_name_or_path)
#         model.register_forward_hook(self.func)
#         return model


class SentenceDebiasBertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForMaskedLM(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class INLPBertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaModel(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        model.encoder.register_forward_hook(self.func)
        return model


# class INLPGPT2Model(_INLPModel):
#     def __new__(self, model_name_or_path, projection_matrix):
#         super().__init__(self, model_name_or_path, projection_matrix)
#         model = transformers.GPT2Model.from_pretrained(model_name_or_path)
#         model.register_forward_hook(self.func)
#         return model


class INLPBertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        model.bert.register_forward_hook(self.func)
        return model


class INLPAlbertForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        model.albert.register_forward_hook(self.func)
        return model


class INLPRobertaForMaskedLM(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        model.roberta.register_forward_hook(self.func)
        return model


class CDABertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class CDAAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class CDARobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model


# class CDAGPT2Model:
#     def __new__(self, model_name_or_path):
#         model = transformers.GPT2Model.from_pretrained(model_name_or_path)
#         return model


class CDABertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDAAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class CDARobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model





class DropoutBertModel:
    def __new__(self, model_name_or_path):
        model = transformers.BertModel.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertModel:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertModel.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaModel:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaModel.from_pretrained(model_name_or_path)
        return model

#
# class DropoutGPT2Model:
#     def __new__(self, model_name_or_path):
#         model = transformers.GPT2Model.from_pretrained(model_name_or_path)
#         return model


class DropoutBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.BertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.AlbertForMaskedLM.from_pretrained(model_name_or_path)
        return model


class DropoutRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = transformers.RobertaForMaskedLM.from_pretrained(model_name_or_path)
        return model


class BertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class AlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class RobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SentenceDebiasBertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasAlbertForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasRobertaForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class SentenceDebiasGPT2ForSequenceClassification(_SentenceDebiasModel):
    def __new__(self, model_name_or_path, bias_direction, config):
        super().__init__(self, model_name_or_path, bias_direction)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class INLPBertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.bert.encoder.register_forward_hook(self.func)
        return model


class INLPAlbertForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.albert.encoder.register_forward_hook(self.func)
        return model


class INLPRobertaForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.roberta.encoder.register_forward_hook(self.func)
        return model


class INLPGPT2ForSequenceClassification(_INLPModel):
    def __new__(self, model_name_or_path, projection_matrix, config):
        super().__init__(self, model_name_or_path, projection_matrix)
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        model.transformer.register_forward_hook(self.func)
        return model


class CDABertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDARobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutBertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.BertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutAlbertForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.AlbertForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutRobertaForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.RobertaForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model


class SelfDebiasBertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasAlbertForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model


class SelfDebiasRobertaForMaskedLM:
    def __new__(self, model_name_or_path):
        model = MaskedLMWrapper(model_name_or_path)
        return model