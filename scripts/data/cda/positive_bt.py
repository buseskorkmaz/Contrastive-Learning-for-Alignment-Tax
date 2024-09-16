import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from difflib import SequenceMatcher
import argparse
from tqdm import tqdm

def load_translation_models():
    model_names = {
        'en_fr': 'Helsinki-NLP/opus-mt-en-fr',
        'en_de': 'Helsinki-NLP/opus-mt-en-de',
        'en_es': 'Helsinki-NLP/opus-mt-en-es',
        'fr_en': 'Helsinki-NLP/opus-mt-fr-en',
        'de_en': 'Helsinki-NLP/opus-mt-de-en',
        'es_en': 'Helsinki-NLP/opus-mt-es-en'
    }
    models = {}
    tokenizers = {}
    for key, name in model_names.items():
        tokenizers[key] = MarianTokenizer.from_pretrained(name)
        models[key] = MarianMTModel.from_pretrained(name)
    return models, tokenizers

def translate(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    translated = model.generate(**inputs, num_beams=5, num_return_sequences=1, temperature=0.8, do_sample=True)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

def backtranslate(text, models, tokenizers, device, src_lang='en', tgt_lang='fr'):
    try:
        intermediate = translate(text, models[f'{src_lang}_{tgt_lang}'], tokenizers[f'{src_lang}_{tgt_lang}'], device)
        result = translate(intermediate, models[f'{tgt_lang}_{src_lang}'], tokenizers[f'{tgt_lang}_{src_lang}'], device)
        return result
    except Exception as e:
        print(f"Error in backtranslation: {e}")
        return text

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_diverse(original, augmented, augmented_list, threshold=0.7):
    return (similarity(original, augmented) < threshold and 
            all(similarity(aug, augmented) < threshold for aug in augmented_list))

def augment_data(source_lines, target_lines, models, tokenizers, device, num_augmentations=3):
    augmented_source_lines = []
    augmented_target_lines = []
    languages = ['fr', 'de', 'es']
    
    for source, target in tqdm(zip(source_lines, target_lines), total=len(source_lines), desc="Augmenting data"):
        augmented_source_lines.append(source.strip())
        augmented_target_lines.append(target.strip())
        
        augmented_pairs = []
        attempts = 0
        while len(augmented_pairs) < num_augmentations and attempts < num_augmentations * 3:
            lang = languages[attempts % len(languages)]
            augmented_source = backtranslate(source, models, tokenizers, device, 'en', lang)
            augmented_target = backtranslate(target, models, tokenizers, device, 'en', lang)
            
            if (is_diverse(source, augmented_source, [pair[0] for pair in augmented_pairs]) and
                is_diverse(target, augmented_target, [pair[1] for pair in augmented_pairs])):
                augmented_pairs.append((augmented_source, augmented_target))
                augmented_source_lines.append(augmented_source)
                augmented_target_lines.append(augmented_target)
            attempts += 1
    
    return augmented_source_lines, augmented_target_lines

def process_files(input_folder, output_folder, models, tokenizers, device, num_augmentations=3):
    for subset in ['train', 'test', 'validation']:
        source_file = os.path.join(input_folder, f'{subset}.source')
        target_file = os.path.join(input_folder, f'{subset}.target')
        
        print(f"Processing {subset} data...")
        with open(source_file, 'r', encoding='utf-8') as src_file, open(target_file, 'r', encoding='utf-8') as tgt_file:
            source_lines = src_file.readlines()
            target_lines = tgt_file.readlines()
        
        augmented_source_lines, augmented_target_lines = augment_data(
            source_lines, target_lines, models, tokenizers, device, num_augmentations
        )
        
        os.makedirs(output_folder, exist_ok=True)
        augmented_source_file = os.path.join(output_folder, f'{subset}.source')
        augmented_target_file = os.path.join(output_folder, f'{subset}.target')
        
        print(f"Saving augmented {subset} data...")
        with open(augmented_source_file, 'w', encoding='utf-8') as aug_src_file, open(augmented_target_file, 'w', encoding='utf-8') as aug_tgt_file:
            for src, tgt in tqdm(zip(augmented_source_lines, augmented_target_lines), total=len(augmented_source_lines), desc=f"Writing {subset} files"):
                aug_src_file.write(f"{src}\n")
                aug_tgt_file.write(f"{tgt}\n")
        
        print(f"Augmented {subset} data saved to {augmented_source_file} and {augmented_target_file}")

def main(args):
    input_folder = args.input_dir
    output_folder = args.output_dir
    num_augmentations = args.num_augmentations
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading translation models...")
    models, tokenizers = load_translation_models()
    for model in models.values():
        model.to(device)
    
    process_files(input_folder, output_folder, models, tokenizers, device, num_augmentations)
    
    print("Augmentation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation using back-translation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing source and target files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for augmented files")
    parser.add_argument("--num_augmentations", type=int, default=3, help="Number of augmented examples to generate per original example")
    
    args = parser.parse_args()
    main(args)