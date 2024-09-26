import os
import json
import pandas as pd
from datetime import datetime

def process_jsonl_file(file_path):
    toxicity_scores = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            toxicity_scores.append(data['toxicity_score'])
    return sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else 0

def main():
    output_dir = '/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output'
    target_date = '20240904'
    results = []

    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl') and target_date in filename:
            model_name = filename.split('_results_')[0]
            file_path = os.path.join(output_dir, filename)
            avg_toxicity = process_jsonl_file(file_path)
            results.append({
                'model': model_name,
                'average_toxicity': avg_toxicity
            })
    
    output_dir = '/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output_base'
    target_date = '20240905'

    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl') and target_date in filename:
            model_name = filename.split('_results_')[0]
            file_path = os.path.join(output_dir, filename)
            avg_toxicity = process_jsonl_file(file_path)
            results.append({
                'model': model_name,
                'average_toxicity': avg_toxicity
            })

    output_dir = '/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output_dropout'
    target_date = '20240906'

    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl') and target_date in filename:
            model_name = filename.split('_results_')[0]
            file_path = os.path.join(output_dir, filename)
            avg_toxicity = process_jsonl_file(file_path)
            results.append({
                'model': model_name,
                'average_toxicity': avg_toxicity
            })
    
    output_dir = '/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output_instructive_debiasing'
    target_date = '20240911'

    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl') and target_date in filename:
            model_name = filename.split('_results_')[0]
            file_path = os.path.join(output_dir, filename)
            avg_toxicity = process_jsonl_file(file_path)
            results.append({
                'model': model_name,
                'average_toxicity': avg_toxicity
            })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    df = df.sort_values('average_toxicity')
    output_file = '/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/realtoxicityprompts/output/toxicity_results_summary_20240911.csv'
    df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    main()