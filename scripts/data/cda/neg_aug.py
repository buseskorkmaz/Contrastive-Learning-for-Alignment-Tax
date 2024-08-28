import argparse
import os
import random
import spacy
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def load_model(model_name):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

def swap_entities_and_numbers(text):
    doc = nlp(text)
    entities_and_numbers = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "CARDINAL", "DATE"]]
    
    if len(entities_and_numbers) < 2:
        return text
    
    swap_indices = random.sample(range(len(entities_and_numbers)), 2)
    a, b = swap_indices
    
    new_text = text
    for i, j in [(a, b), (b, a)]:
        new_text = new_text.replace(str(entities_and_numbers[i]), f"__{i}__", 1)
    
    for i in [a, b]:
        new_text = new_text.replace(f"__{i}__", str(entities_and_numbers[i]), 1)
    
    return new_text

def compute_confidence(model, tokenizer, text, device):
    if not text.strip():  # Check if the text is empty or just whitespace
        return 0.0  # Return a low confidence score for empty inputs
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    if inputs['input_ids'].numel() == 0:  # Check if the tokenized input is empty
        return 0.0
    
    with torch.no_grad():
        try:
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            first_layer = hidden_states[1]
            last_layer = hidden_states[-1]
            similarity = torch.nn.functional.cosine_similarity(first_layer, last_layer)
            confidence = similarity.mean().item()
        except RuntimeError as e:
            print(f"Error processing text: {text}")
            print(f"Error message: {str(e)}")
            return 0.0  # Return a low confidence score if there's an error
    
    return confidence

def select_low_confidence(model, tokenizer, generations, device, threshold=0.5):
    low_conf_gens = []
    for gen in generations:
        try:
            confidence = compute_confidence(model, tokenizer, gen, device)
            if confidence < threshold:
                low_conf_gens.append(gen)
        except Exception as e:
            print(f"Error processing generation: {gen}")
            print(f"Error message: {str(e)}")
    return low_conf_gens

def masked_regeneration(model, tokenizer, text, device, mask_probability=0.15, topk=5):
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    if not words:
        return [text]  # Return original text if no words to mask
    
    num_to_mask = max(1, int(len(words) * mask_probability))
    indices_to_mask = random.sample(range(len(words)), num_to_mask)
    
    masked_texts = []
    for idx in indices_to_mask:
        masked_words = words.copy()
        original_word = masked_words[idx]
        masked_words[idx] = tokenizer.mask_token
        masked_text = ' '.join(masked_words)
        masked_texts.append((masked_text, idx, original_word))
    
    new_texts = []
    for masked_text, mask_idx, original_word in masked_texts:
        input_ids = tokenizer.encode(masked_text, return_tensors='pt').to(device)
        mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        
        if len(mask_token_index) == 0:
            print(f"Warning: Mask token not found in: {masked_text}")
            continue
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        
        mask_token_logits = logits[0, mask_token_index, :]
        top_tokens = torch.topk(mask_token_logits, topk, dim=1).indices[0].tolist()
        
        for token in top_tokens:
            new_word = tokenizer.decode([token])
            if new_word and new_word.strip() and new_word.strip() != original_word:
                new_words = words.copy()
                new_words[mask_idx] = new_word.strip()
                new_text = ' '.join(word for word in new_words if word)  # Filter out any None or empty strings
                if new_text != text:  # Only add if the new text is different from the original
                    new_texts.append(new_text)
    
    return new_texts if new_texts else [text]  # Return original text if no augmentations were generated

def process_data(input_dir, output_dir, strategies, model, tokenizer, device):
    for split in ['train', 'test', 'validation']:
        source_file = os.path.join(input_dir, f'{split}.source')
        target_file = os.path.join(input_dir, f'{split}.target')
        
        with open(source_file, 'r') as src_f, open(target_file, 'r') as tgt_f:
            sources = src_f.readlines()
            targets = tgt_f.readlines()
        
        assert len(sources) == len(targets), f"{split} files must have the same number of lines"
        
        for strategy in strategies:
            strategy_dir = os.path.join(output_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
            
            aug_sources, aug_targets = [], []
            
            for source, target in tqdm(zip(sources, targets), desc=f"Applying {strategy} to {split}"):
                source, target = source.strip(), target.strip()
                
                if strategy == 'entity_swap':
                    aug_source = swap_entities_and_numbers(source)
                    aug_target = swap_entities_and_numbers(target)
                    aug_sources.append(aug_source)
                    aug_targets.append(aug_target)
                
                elif strategy == 'low_confidence':
                    generations = [gen.strip() for gen in target.split('|') if gen.strip()]  # Remove empty generations
                    if generations:
                        low_conf_gens = select_low_confidence(model, tokenizer, generations, device)
                        if low_conf_gens:
                            aug_sources.append(source)
                            aug_targets.append('|'.join(low_conf_gens))
                    else:
                        print(f"Warning: Empty generations for source: {source}")
                
                elif strategy == 'masked_regen':
                    try:
                        new_targets = masked_regeneration(model, tokenizer, target, device)
                        for new_target in new_targets:
                            if new_target != target:  # Only add if the augmentation is different from the original
                                aug_sources.append(source)
                                aug_targets.append(new_target)
                        if not new_targets or all(nt == target for nt in new_targets):
                            print(f"No valid augmentation generated for: {target}")
                    except Exception as e:
                        print(f"Error in masked regeneration for target: {target}")
                        print(f"Error message: {str(e)}")

            # Write augmented data
            with open(os.path.join(strategy_dir, f'{split}.source'), 'w') as f:
                f.writelines(f"{line}\n" for line in aug_sources)
            with open(os.path.join(strategy_dir, f'{split}.target'), 'w') as f:
                f.writelines(f"{line}\n" for line in aug_targets)

def main():
    parser = argparse.ArgumentParser(description="Negative data augmentation with GPT-2")
    parser.add_argument('--input_dir', help="Input directory containing source, target, and other files", 
                        default="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/neg")
    parser.add_argument('--output_dir', help="Output directory for augmented data", 
                        default="/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/neg_processed")
    parser.add_argument('--model_name', default='openai-community/gpt2-xl', help="Hugging Face model name")
    parser.add_argument('--strategies', nargs='+', default=['masked_regen'],
                        choices=['entity_swap', 'low_confidence', 'masked_regen'],
                        help="Augmentation strategies to apply")
    parser.add_argument('--topk', type=int, default=5, help="Top k predictions for masked regeneration")
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help="Threshold for low confidence selection")
    
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model_name)
    model.eval()
    
    process_data(args.input_dir, args.output_dir, args.strategies, model, tokenizer, device)

if __name__ == '__main__':
    main()