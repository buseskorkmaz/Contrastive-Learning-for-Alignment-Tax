import os
import sys
import argparse
import re
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
import pandas as pd
from src.atomic_facts import AtomicFactGenerator

def preprocess_text(text):
    # Remove the part starting with "Based on the original" and everything after
    pattern = r"Based on the original.*$"
    cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

def main(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)
    df['evaluated_text'] = df['evaluated_text'].fillna('empty')

    # Preprocess the evaluated_text
    df['evaluated_text'] = df['evaluated_text'].apply(preprocess_text)

    # Create an AtomicFactGenerator instance
    generator = AtomicFactGenerator(model='meta-llama/llama-3-70b-instruct')

    # Function to extract atomic facts from a column
    def extract_atomic_facts(texts):
        return generator.run(texts)

    # Extract atomic facts for the generated dataset
    df['atomic_facts'] = extract_atomic_facts(df['evaluated_text'].tolist())

    # Save the updated dataset
    df.to_csv(output_file, index=False)

    # Optionally, you can print some samples to compare
    print("Sample comparison:")
    for i in range(min(5, len(df))):
        print(f"\nRow {i}:")
        print("Text:", df['evaluated_text'].iloc[i])
        print("Atomic facts:", df['atomic_facts'].iloc[i])
        print("***"*50, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract atomic facts from dataset")
    parser.add_argument("--input_file", help="Path to the input CSV file")
    parser.add_argument("--output_file", default="dataset_with_atomic_facts.csv", 
                        help="Path to the output CSV file (default: dataset_with_atomic_facts.csv)")
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)