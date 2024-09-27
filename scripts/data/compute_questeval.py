import pandas as pd
import numpy as np
import os
import sys
import re
import argparse
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
# from src.factuality_detector import FactualityDetector
import nltk
nltk.download('punkt')
from QuestEval.questeval.questeval_metric import QuestEval

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


def calculate_questeval_scores(original_texts, generated_texts):
    questeval = QuestEval()
    scores = questeval.corpus_questeval(
        hypothesis=generated_texts,
        sources=original_texts
    )
    return scores['corpus_score'], scores['ex_level_scores']

def process_row(row, questeval):
    original_text = row['original_text']
    generated_text = row['generated_text']
    
    original_sentences = split_into_sentences(original_text)
    generated_sentences = split_into_sentences(generated_text)
    
    if not generated_sentences:
        return pd.DataFrame({
            'original_text': [original_text],
            'generated_text': [generated_text],
            'generated_sentence': [''],
            # 'factuality_score': [0.0],
            # 'similarity_score': [0.0],
            # 'most_similar_original_sentence': [''],
            'questeval_score': [0.0]
        })
    
    # factuality_scores = create_factuality_matrix(original_text, generated_sentences, detector)
    # similarity_matrix = create_similarity_matrix(original_sentences, generated_sentences)
    
    # highest_similarity_scores = np.max(similarity_matrix, axis=1)
    # most_similar_original_indices = np.argmax(similarity_matrix, axis=1)
    # most_similar_original_sentences = [original_sentences[i] for i in most_similar_original_indices]
    
    # Calculate QuestEval scores for each generated sentence
    questeval_scores = [
        questeval.corpus_questeval(hypothesis=[sentence], sources=[original_text])['corpus_score']
        for sentence in generated_sentences
    ]
    
    return pd.DataFrame({
        'original_text': [original_text] * len(generated_sentences),
        'generated_text': [generated_text] * len(generated_sentences),
        'generated_sentence': generated_sentences,
        # 'factuality_score': factuality_scores,
        # 'similarity_score': highest_similarity_scores,
        # 'most_similar_original_sentence': most_similar_original_sentences,
        'questeval_score': questeval_scores
    })

def main(args):
    # Load data
    original_df = pd.read_csv("/gpfs/home/bsk18/factual-bias-mitigation/data/input_files/hackernews_original_eval_df.csv")
    generated_df = pd.read_csv("/gpfs/home/bsk18/factual-bias-mitigation/data/input_files/hackernews_generated_eval_df.csv")
    
    # Print info about the dataframes
    print("Original DataFrame info:")
    print(original_df.info())
    print("\nGenerated DataFrame info:")
    print(generated_df.info())

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
    # global model
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")
    questeval = QuestEval()

    # Calculate QuestEval scores for entire texts
    print("Calculating QuestEval scores for entire texts...")
    overall_questeval_score, individual_questeval_scores = calculate_questeval_scores(
        input_df['original_text'].tolist(),
        input_df['generated_text'].tolist()
    )

    # Add QuestEval scores to input_df
    input_df['questeval_score'] = individual_questeval_scores

    # Process each row with tqdm progress bar
    tqdm.pandas(desc="Processing rows")
    result = input_df.progress_apply(lambda row: process_row(row, questeval), axis=1)
    
    # Concatenate all results
    output_df = pd.concat(result.tolist(), ignore_index=True)

    # Save results for sentence-level analysis
    sentence_output_file = "/gpfs/home/bsk18/factual-bias-mitigation/data/output_files/hackernews_sentence_level_analysis_questeval.csv"
    output_df.to_csv(sentence_output_file, index=False)
    print(f"Sentence-level results saved to {sentence_output_file}")

    # Save results for text-level analysis
    text_output_file = "/gpfs/home/bsk18/factual-bias-mitigation/data/output_files/hackernews_text_level_analysis_questeval.csv"
    text_level_df = input_df.copy()
    text_level_df['overall_questeval_score'] = overall_questeval_score
    text_level_df.to_csv(text_output_file, index=False)
    print(f"Text-level results saved to {text_output_file}")

    # Print first few rows as demo if requested
    if args.print_demo:
        print("\nSentence-level Demo (First 5 Rows):")
        print(output_df[['original_text', 'generated_text', 'generated_sentence', 'factuality_score', 'similarity_score', 'most_similar_original_sentence', 'questeval_score']].head())
        print("\nText-level Demo (First 5 Rows):")
        print(text_level_df[['original_text', 'generated_text', 'questeval_score', 'overall_questeval_score']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze factuality, similarity, and QuestEval scores of generated sentences and texts.")
    parser.add_argument("--print_demo", action="store_true", help="Print the first few rows as a demo")
    args = parser.parse_args()
    main(args)