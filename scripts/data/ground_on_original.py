import pandas as pd
import numpy as np
import os
import sys
import re
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from src.factuality_detector import FactualityDetector
import nltk
nltk.download('punkt')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string inputs
    pattern = r"Based on the original.*$"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip().lower()

def split_into_sentences(text):
    if not isinstance(text, str) or not text.strip():
        return ['']  # Return a list with an empty string if input is invalid
    sentences = nltk.sent_tokenize(text)
    return sentences if sentences else ['']  # Return a list with an empty string if no sentences were found

def create_similarity_matrix(original_sentences, generated_sentences):
    original_embeddings = model.encode(original_sentences)
    generated_embeddings = model.encode(generated_sentences)
    return cosine_similarity(generated_embeddings, original_embeddings)

def create_factuality_matrix(original_text, generated_sentences, detector):
    factuality_scores = []
    for sentence in generated_sentences:
        score = safe_generate_score(detector, original_text, sentence)
        factuality_scores.append(score)
    return np.array(factuality_scores)

def safe_generate_score(detector, orig_text, gen_text, n_gram=2, n_skip=1):
    try:
        _, _, score = detector.generate_score([orig_text], [gen_text], summac_style=True, n_gram=n_gram, n_skip=n_skip)
        return score
    except Exception as e:
        print(f"Error in generate_score: {e}")
        print(f"Original text: {orig_text}")
        print(f"Generated text: {gen_text}")
        return 0.0

def process_row(row, detector):
    original_text = row['original_text']
    generated_text = row['generated_text']
    
    original_sentences = split_into_sentences(original_text)
    generated_sentences = split_into_sentences(generated_text)
    
    if not generated_sentences:
        # Return a DataFrame with a single row of empty/zero values
        return pd.DataFrame({
            'original_text': [original_text],
            'generated_text': [generated_text],
            'generated_sentence': [''],
            'factuality_score': [0.0],
            'similarity_score': [0.0],
            'most_similar_original_sentence': ['']
        })
    
    factuality_scores = create_factuality_matrix(original_text, generated_sentences, detector)
    similarity_matrix = create_similarity_matrix(original_sentences, generated_sentences)
    
    highest_similarity_scores = np.max(similarity_matrix, axis=1)
    most_similar_original_indices = np.argmax(similarity_matrix, axis=1)
    most_similar_original_sentences = [original_sentences[i] for i in most_similar_original_indices]
    
    return pd.DataFrame({
        'original_text': [original_text] * len(generated_sentences),
        'generated_text': [generated_text] * len(generated_sentences),
        'generated_sentence': generated_sentences,
        'factuality_score': factuality_scores,
        'similarity_score': highest_similarity_scores,
        'most_similar_original_sentence': most_similar_original_sentences
    })

def main(args):
    # Load data
    original_df = pd.read_csv("/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/workable_original_eval_df.csv")
    generated_df = pd.read_csv("/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/input_files/workable_generated_eval_df.csv")
    
    # Print info about the dataframes
    print("Original DataFrame info:")
    print(original_df.info())
    print("\nGenerated DataFrame info:")
    print(generated_df.info())

    # Print the first few rows of each dataframe
    print("\nFirst few rows of original_df:")
    print(original_df.head())
    print("\nFirst few rows of generated_df:")
    print(generated_df.head())

    # Preprocess texts
    print("\nPreprocessing original texts...")
    original_df['original_text'] = original_df['evaluated_text'].apply(preprocess_text)
    print("Preprocessing generated texts...")
    generated_df['generated_text'] = generated_df['evaluated_text'].apply(preprocess_text)

    # Prepare input dataframe
    input_df = pd.DataFrame({
        'original_text': original_df['original_text'],
        'generated_text': generated_df['generated_text']
    })

    # Initialize models
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")

    # Process each row with tqdm progress bar
    tqdm.pandas(desc="Processing rows")
    result = input_df.progress_apply(lambda row: process_row(row, detector), axis=1)
    
    # Concatenate all results
    output_df = pd.concat(result.tolist(), ignore_index=True)

    # Save results
    output_file = "/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/output_files/workable_sentence_level_factuality_analysis.csv"
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print first few rows as demo if requested
    if args.print_demo:
        print("\nDemo (First 5 Rows):")
        print(output_df[['original_text', 'generated_text', 'generated_sentence', 'factuality_score', 'similarity_score', 'most_similar_original_sentence']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze factuality and similarity of generated sentences.")
    parser.add_argument("--print_demo", action="store_true", help="Print the first few rows as a demo")
    args = parser.parse_args()
    main(args)