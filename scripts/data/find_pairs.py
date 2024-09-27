import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import sys
import re
import argparse
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from src.factuality_detector import FactualityDetector

def avg_length_of_factoids(factoids_df):
    generated_length = sum(len(row) for row in factoids_df['generated_facts'])
    original_length = sum(len(row) for row in factoids_df['original_facts'])
    print("Original: ", original_length/len(factoids_df), "Generated: ", generated_length/len(factoids_df))

def remove_duplicate_facts(row):
    original = set(fact.lower().replace(',', '').replace('.', '').strip() for fact in row['original_facts'])
    generated = set(fact.lower().replace(',', '').replace('.', '').strip() for fact in row['generated_facts'])
    
    unique_original = [fact for fact in row['original_facts'] if fact.lower().replace(',', '').replace('.', '').strip() not in generated]
    unique_generated = [fact for fact in row['generated_facts'] if fact.lower().replace(',', '').replace('.', '').strip() not in original]
    
    return pd.Series({
        'original_facts': unique_original,
        'generated_facts': unique_generated
    })

def create_similarity_matrix(original_facts, generated_facts):
    if not original_facts or not generated_facts:
        return np.array([])
    original_embeddings = model.encode(original_facts)
    generated_embeddings = model.encode(generated_facts)
    return cosine_similarity(original_embeddings, generated_embeddings)

def clean_facts(facts):
    return [fact for fact in facts if isinstance(fact, str) and fact.strip() and 'redacted_email' not in fact.lower() and 'http' not in fact.lower()]

def safe_generate_score(detector, orig_fact, gen_fact, n_gram=2, n_skip=1):
    try:
        _, _, score = detector.generate_score([orig_fact], [gen_fact], summac_style=True, n_gram=n_gram, n_skip=n_skip)
        return score
    except Exception as e:
        print(f"Error in generate_score: {e}")
        print(f"Original fact: {orig_fact}")
        print(f"Generated fact: {gen_fact}")
        return 0.0

def create_factuality_matrix(original_facts, generated_facts, detector, n_gram=2, n_skip=1):
    if not original_facts or not generated_facts:
        return np.array([])
    factuality_matrix = np.zeros((len(original_facts), len(generated_facts)))
    for i, orig_fact in enumerate(original_facts):
        for j, gen_fact in enumerate(generated_facts):
            factuality_matrix[i, j] = safe_generate_score(detector, orig_fact, gen_fact, n_gram, n_skip)
    return factuality_matrix

def process_row(row, detector):
    original_facts = row['original_facts']
    generated_facts = row['generated_facts']
    
    if not original_facts or not generated_facts:
        return pd.Series({
            'most_relevant_evidence': [],
            'most_similar_facts': [],
            'factuality_scores': [],
            'similarity_scores': []
        })
    
    factuality_matrix = create_factuality_matrix(original_facts, generated_facts, detector)
    similarity_matrix = create_similarity_matrix(original_facts, generated_facts)
    
    most_relevant_evidence = []
    most_similar_facts = []
    factuality_scores = []
    similarity_scores = []
    
    for j in range(len(generated_facts)):
        max_factuality_index = np.argmax(factuality_matrix[:, j])
        max_similarity_index = np.argmax(similarity_matrix[:, j])
        
        most_relevant_evidence.append(original_facts[max_factuality_index])
        most_similar_facts.append(original_facts[max_similarity_index])
        factuality_scores.append(factuality_matrix[max_factuality_index, j])
        similarity_scores.append(similarity_matrix[max_similarity_index, j])
    
    return pd.Series({
        'most_relevant_evidence': most_relevant_evidence,
        'most_similar_facts': most_similar_facts,
        'factuality_scores': factuality_scores,
        'similarity_scores': similarity_scores
    })

def main(args):
    # Load data
    original_df = pd.read_csv("/gpfs/home/bsk18/factual-bias-mitigation/data/output_files/hackernews_original_eval_df_with_facts.csv")
    generated_df = pd.read_csv("/gpfs/home/bsk18/factual-bias-mitigation/data/output_files/hackernews_generated_eval_df_with_facts.csv")

    # Prepare factoids_df
    factoids_df = pd.DataFrame({
        'original_text': original_df['evaluated_text'],
        'generated_text': generated_df['evaluated_text'],
        'original_facts': original_df['atomic_facts'].str.replace('[', '').str.replace(']', '').str.split(',').apply(lambda x: [y.strip("' ") for y in x if isinstance(y, str)]).apply(clean_facts),
        'generated_facts': generated_df['atomic_facts'].str.replace('[', '').str.replace(']', '').str.split(',').apply(lambda x: [y.strip("' ") for y in x if isinstance(y, str)]).apply(clean_facts)
    })
    print("Average number of factoids per row: ")
    avg_length_of_factoids(factoids_df)

    # Remove duplicate facts
    factoids_df[['original_facts', 'generated_facts']] = factoids_df.apply(remove_duplicate_facts, axis=1)
    print("Average number of factoids per row after dropping exact facts in the original and generated: ")
    avg_length_of_factoids(factoids_df)

    # Initialize models
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")

    # Process each row with tqdm progress bar
    tqdm.pandas(desc="Processing rows")
    result = factoids_df.progress_apply(lambda row: process_row(row, detector), axis=1)
    
    # Add new columns to factoids_df
    factoids_df['most_relevant_evidence'] = result['most_relevant_evidence']
    factoids_df['most_similar_facts'] = result['most_similar_facts']
    factoids_df['factuality_scores'] = result['factuality_scores']
    factoids_df['similarity_scores'] = result['similarity_scores']

    # Save results
    output_file = "/gpfs/home/bsk18/factual-bias-mitigation/data/output_files/hackernews_factuality_similarity_analysis.csv"
    factoids_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Print first row as demo if requested
    if args.print_demo:
        print("\nDemo (First Row):")
        row = factoids_df.iloc[0]
        for i, (gen_fact, rel_evidence, sim_fact, fact_score, sim_score) in enumerate(zip(
            row['generated_facts'], row['most_relevant_evidence'], row['most_similar_facts'],
            row['factuality_scores'], row['similarity_scores']
        )):
            print(f"\nGenerated Fact {i+1}: {gen_fact}")
            print(f"Most Relevant Evidence (Factuality Score: {fact_score:.4f}):")
            print(f"  {rel_evidence}")
            print(f"Most Similar Fact (Similarity Score: {sim_score:.4f}):")
            print(f"  {sim_fact}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze factuality and similarity of generated facts.")
    parser.add_argument("--print_demo", action="store_true", help="Print the first row as a demo")
    args = parser.parse_args()
    main(args)