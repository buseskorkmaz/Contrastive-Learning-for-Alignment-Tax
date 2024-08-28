import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from src.factuality_detector import FactualityDetector

def safe_generate_score(detector, orig_text, gen_text, n_gram=2, n_skip=1):
    try:
        _, _, score = detector.generate_score([orig_text], [gen_text], summac_style=True, n_gram=n_gram, n_skip=n_skip)
        return score
    except Exception as e:
        print(f"Error in generate_score: {e}")
        print(f"Original text: {orig_text}")
        print(f"Generated text: {gen_text}")
        return 0.0

def main():
    # Load the data
    df = pd.read_csv("/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/output_files/workable_sentence_level_factuality_analysis.csv")

    # Initialize the factuality detector
    detector = FactualityDetector("/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli")

    # Group the dataframe by original_text and generated_text
    grouped = df.groupby(['original_text', 'generated_text'])

    # Initialize lists to store results
    original_texts = []
    original_generated_texts = []
    fixed_generated_texts = []
    original_factuality_scores = []
    new_factuality_scores = []

    # Process each group
    for (original_text, original_generated_text), group in tqdm(grouped):
        fixed_generated_text = original_generated_text
        sentences_to_replace = group[(group['factuality_score'] < 0.1) & (group['similarity_score'] >= 0.4)]

        for _, row in sentences_to_replace.iterrows():
            fixed_generated_text = fixed_generated_text.replace(row['generated_sentence'], row['most_similar_original_sentence'])

        # Calculate original and new factuality scores
        original_score = safe_generate_score(detector, original_text, original_generated_text)
        new_score = safe_generate_score(detector, original_text, fixed_generated_text)

        # Store results
        original_texts.append(original_text)
        original_generated_texts.append(original_generated_text)
        fixed_generated_texts.append(fixed_generated_text)
        original_factuality_scores.append(original_score)
        new_factuality_scores.append(new_score)

        # Log the fixed generated text
        print(f"Original text: {original_text}")
        print(f"Original generated text: {original_generated_text}")
        print(f"Fixed generated text: {fixed_generated_text}")
        print(f"Original factuality score: {original_score}")
        print(f"New factuality score: {new_score}")
        print("---")

    # Create a dataframe with the results
    results_df = pd.DataFrame({
        'original_text': original_texts,
        'original_generated_text': original_generated_texts,
        'fixed_generated_text': fixed_generated_texts,
        'original_factuality_score': original_factuality_scores,
        'new_factuality_score': new_factuality_scores
    })

    # Save the results
    results_df.to_csv("/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/output_files/workable_factuality_improvement_results.csv", index=False)

    # Print summary statistics
    print("Summary Statistics:")
    print(f"Average original factuality score: {np.mean(original_factuality_scores)}")
    print(f"Average new factuality score: {np.mean(new_factuality_scores)}")
    print(f"Improvement: {np.mean(new_factuality_scores) - np.mean(original_factuality_scores)}")

if __name__ == "__main__":
    main()