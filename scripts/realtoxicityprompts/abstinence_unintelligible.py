import os
import json
import pandas as pd
from collections import defaultdict
import re

def get_method_from_filename(filename):
    """Extract method from filename."""
    if 'cda' in filename:
        if 'gender' in filename:
            return 'cda_gender'
        elif 'race' in filename:
            return 'cda_race'
        elif 'religion' in filename:
            return 'cda_religion'
        return 'cda'
    elif 'inlp' in filename:
        if 'race' in filename:
            return 'inlp_race'
        return 'inlp'
    elif 'dropout' in filename:
        return 'dropout'
    elif 'instructive_debiasing' in filename:
        return 'instructive'
    elif 'self_debiasing' in filename:
        return 'self'
    elif 'sentence_debiasing' in filename:
        if 'race' in filename:
            return 'sentence_race'
        return 'sentence'
    else:
        return 'base'

def get_model_from_filename(filename):
    """Extract model name from filename."""
    if 'llama2' in filename.lower():
        return 'llama2-7b'
    elif 'phi2' in filename.lower():
        return 'phi2'
    elif 'gpt2' in filename.lower():
        return 'gpt2'
    return 'gpt2'

def analyze_generations(file_path):
    stats = defaultdict(int)
    total_generations = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            continuation = data['continuation'].strip().lower()
            total_generations += 1
            
            # Check for abstention patterns
            if not continuation or continuation.isspace():
                stats['empty_response'] += 1
            elif any(phrase in continuation for phrase in [
                'i am not sure',
                'i cannot',
                'i\'m not sure',
                'i don\'t know',
                'i would rather not',
                'i apologize',
                'i prefer not'
            ]):
                stats['explicit_abstention'] += 1
                
            # Check for unintelligible patterns
            words = continuation.split()
            # Repetitive text
            if len(words) >= 4:
                for i in range(len(words)-3):
                    if words[i:i+2] == words[i+2:i+4]:
                        stats['repetitive_text'] += 1
                        break
            
            # Grammatically incomplete
            if len(words) < 3 or not re.search(r'[.!?]$', continuation):
                stats['incomplete_sentence'] += 1
            
            # Non-sequiturs
            if continuation.startswith(('the', 'a', 'an')) and len(data['prompt'].split()) > 3:
                stats['potential_non_sequitur'] += 1
            
            # Excessive punctuation
            if re.search(r'[.!?]{2,}', continuation) or re.search(r'[,;]{3,}', continuation):
                stats['excessive_punctuation'] += 1
    
    # Calculate percentages
    if total_generations > 0:
        percentages = {
            'empty_responses': (stats['empty_response']/total_generations * 100),
            'explicit_abstention': (stats['explicit_abstention']/total_generations * 100),
            'repetitive_text': (stats['repetitive_text']/total_generations * 100),
            'incomplete_sentences': (stats['incomplete_sentence']/total_generations * 100),
            'potential_non_sequiturs': (stats['potential_non_sequitur']/total_generations * 100),
            'excessive_punctuation': (stats['excessive_punctuation']/total_generations * 100)
        }
    else:
        percentages = {k: 0.0 for k in ['empty_responses', 'explicit_abstention', 'repetitive_text', 
                                      'incomplete_sentences', 'potential_non_sequiturs', 'excessive_punctuation']}
    
    return percentages

def main():
    # Define all your directories
    directories = [
        'factual-bias-mitigation/scripts/realtoxicityprompts/output',
        'factual-bias-mitigation/scripts/realtoxicityprompts/output_base',
        'factual-bias-mitigation/scripts/realtoxicityprompts/output_dropout',
        'factual-bias-mitigation/scripts/realtoxicityprompts/output_instructive_debiasing'
    ]
    
    all_results = []
    
    # Process all jsonl files in all directories
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.jsonl'):
                    file_path = os.path.join(directory, filename)
                    model = get_model_from_filename(filename)
                    method = get_method_from_filename(filename)
                    
                    if model != 'unknown':
                        results = analyze_generations(file_path)
                        results['model'] = model
                        results['method'] = method
                        all_results.append(results)
                        print(f"Processed {model} with {method} method")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[t]\n\\small\n\\centering\n"
    latex_table += "\\begin{tabular}{l|cc|cccc}\n\\toprule\n"
    latex_table += "\\multirow{2}{*}{Model} & \\multicolumn{2}{c|}{Abstention Patterns (\\%)} & \\multicolumn{4}{c}{Unintelligible Patterns (\\%)} \\\\\n"
    latex_table += " & Empty & Explicit & Repetitive & Incomplete & Non-seq. & Punct. \\\\\n\\midrule\n"
    
    # Add rows for each model and method
    for model in sorted(df['model'].unique()):
        model_data = df[df['model'] == model]
        # First write base model results
        base_data = model_data[model_data['method'] == 'base']
        if not base_data.empty:
            row = base_data.iloc[0]
            latex_table += f"{model} & "
            latex_table += f"{row['empty_responses']:.1f} & {row['explicit_abstention']:.1f} & "
            latex_table += f"{row['repetitive_text']:.1f} & {row['incomplete_sentences']:.1f} & "
            latex_table += f"{row['potential_non_sequiturs']:.1f} & {row['excessive_punctuation']:.1f} \\\\\n"
        
        # Then write other methods
        methods_order = ['inlp', 'inlp_race', 'dropout', 'sentence', 'sentence_race', 'self', 'instructive', 
                        'cda_gender', 'cda_race', 'cda_religion']
        
        for method in methods_order:
            method_data = model_data[model_data['method'] == method]
            if not method_data.empty:
                row = method_data.iloc[0]
                latex_table += f"\\quad + {method} & "
                latex_table += f"{row['empty_responses']:.1f} & {row['explicit_abstention']:.1f} & "
                latex_table += f"{row['repetitive_text']:.1f} & {row['incomplete_sentences']:.1f} & "
                latex_table += f"{row['potential_non_sequiturs']:.1f} & {row['excessive_punctuation']:.1f} \\\\\n"
        
        latex_table += "\\midrule\n"
    
    latex_table = latex_table.rstrip("\\midrule\n") + "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Analysis of generation patterns across different models and debiasing methods. Numbers show percentage of generations exhibiting each pattern. Lower percentages indicate better quality.}\n"
    latex_table += "\\label{tab:generation-patterns}\n\\end{table}"
    
    # Save LaTeX table
    output_path = 'generation_patterns_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table has been generated and saved to '{output_path}'")
    
    # Also save raw data for verification
    df.to_csv('generation_patterns_raw.csv', index=False)
    print(f"Raw data saved to 'generation_patterns_raw.csv'")

if __name__ == "__main__":
    main()