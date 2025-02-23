import os
import pandas as pd
import numpy as np

def get_model_metrics(df, model_prefix):
    """Extract metrics for a specific model."""
    metrics = {}
    try:
        metrics['BLEU'] = df[f'{model_prefix} bleu max'].mean()
        metrics['ROUGE-1'] = df[f'{model_prefix} rouge1 max'].mean()
        metrics['MC1'] = df[f'{model_prefix} MC1'].mean()
        metrics['MC2'] = df[f'{model_prefix} MC2'].mean()
    except:
        return None
    return metrics

def cap_percentage_change(value, max_cap=200):
    """Cap percentage changes to reasonable values."""
    if pd.isna(value) or abs(value) > max_cap:
        return 0.0
    return value

def calculate_percentage_change(new_value, base_value, max_cap=200):
    """Calculate percentage change with safeguards."""
    if base_value == 0 or pd.isna(new_value) or pd.isna(base_value):
        return 0.0
    pct_change = ((new_value - base_value) / abs(base_value)) * 100
    return cap_percentage_change(pct_change, max_cap)

def get_category_group(category):
    """Group categories into broader types."""
    groups = {
        'Factual': ['Science', 'Economics', 'Weather', 'Statistics', 'Health', 'Nutrition'],
        'Reasoning': ['Logical Falsehood', 'Education', 'Finance'],
        'Social': ['Sociology', 'Psychology', 'Law', 'Politics', 'Stereotypes'],
        'Cultural': ['Religion', 'Myths and Fairytales', 'Proverbs', 'Language'],
        'Errors': ['Indexical Error: Identity', 'Indexical Error: Time', 'Indexical Error: Location', 
                  'Indexical Error: Other', 'Confusion: People', 'Confusion: Places', 'Confusion: Other'],
        'Beliefs': ['Conspiracies', 'Paranormal', 'Superstitions', 'Misconceptions'],
        'Other': ['Fiction', 'History', 'Advertising', 'Misinformation', 'Distraction', 'Subjective']
    }
    
    for group, categories in groups.items():
        if category in categories:
            return group
    return 'Other'

def analyze_category_performance(base_dir):
    """Analyze performance changes by category for each model and method."""
    
    # Define model configurations
    model_configs = {
        'GPT2': {
            'base': ('gpt2', 'gpt2'),
            'variants': [
                ('gpt2_cda_gender', 'CDA (gender)'),
                ('gpt2_cda_race', 'CDA (race)'),
                ('gpt2_cda_religion', 'CDA (religion)'),
                ('gpt2_dropout', 'Dropout'),
                ('inlp-race', 'INLP (race)'),
                ('sentence_debiasing-race', 'Sentence (race)')
            ]
        },
        'Llama2-7B': {
            'base': ('llama2-7b', 'llama2-7b'),
            'variants': [
                ('llama2-7b_cda_gender', 'CDA (gender)'),
                ('llama2-7b_cda_race', 'CDA (race)'),
                ('llama2-7b_cda_religion', 'CDA (religion)'),
                ('llama2-7b_dropout', 'Dropout'),
                ('self_debiasing-llama2', 'Self-Debiasing'),
                ('instructive_debiasing-llama2', 'Instructive')
            ]
        },
        'Phi2': {
            'base': ('phi2', 'phi2'),
            'variants': [
                ('phi2_cda_gender', 'CDA (gender)'),
                ('phi2_cda_race', 'CDA (race)'),
                ('phi2_cda_religion', 'CDA (religion)'),
                ('phi2_dropout', 'Dropout'),
                ('self_debiasing-phi2', 'Self-Debiasing'),
                ('instructive_debiasing-phi2', 'Instructive')
            ]
        }
    }
    
    category_results = {}
    
    # Process each model and its variants
    for model_name, config in model_configs.items():
        base_folder = os.path.join(base_dir, config['base'][0])
        if not os.path.exists(base_folder):
            continue
            
        base_file = os.path.join(base_folder, 'answers.csv')
        if not os.path.exists(base_file):
            continue
            
        base_df = pd.read_csv(base_file)
        base_categories = {}
        
        for category in base_df['Category'].unique():
            category_df = base_df[base_df['Category'] == category]
            base_metrics = get_model_metrics(category_df, config['base'][1])
            if base_metrics:
                base_categories[category] = base_metrics
        
        # Process variants
        for variant_path, variant_name in config['variants']:
            variant_folder = os.path.join(base_dir, variant_path)
            if not os.path.exists(variant_folder):
                continue
                
            variant_file = os.path.join(variant_folder, 'answers.csv')
            if not os.path.exists(variant_file):
                continue
                
            variant_df = pd.read_csv(variant_file)
            
            for category in variant_df['Category'].unique():
                if category not in category_results:
                    category_results[category] = {
                        'changes': [],
                        'count': len(variant_df[variant_df['Category'] == category]),
                        'group': get_category_group(category)
                    }
                
                category_df = variant_df[variant_df['Category'] == category]
                variant_metrics = get_model_metrics(category_df, variant_path)
                
                if variant_metrics and category in base_categories:
                    changes = {
                        metric: calculate_percentage_change(variant_metrics[metric], 
                                                         base_categories[category][metric])
                        for metric in variant_metrics
                    }
                    category_results[category]['changes'].append(changes)
    
    # Calculate average changes
    final_results = {}
    for category, data in category_results.items():
        if not data['changes']:
            continue
            
        avg_changes = {}
        for metric in ['BLEU', 'ROUGE-1', 'MC1', 'MC2']:
            values = [change[metric] for change in data['changes'] if metric in change]
            avg_changes[metric] = np.mean(values) if values else 0.0
        
        final_results[category] = {
            'metrics': avg_changes,
            'count': data['count'],
            'group': data['group']
        }
    
    # Generate LaTeX table
    latex_table = "\\begin{table}[t]\n\\small\n\\centering\n"
    latex_table += "\\begin{tabular}{l|l|cccc|r}\n\\toprule\n"
    latex_table += "Group & Category & BLEU & ROUGE-1 & MC1 & MC2 & N \\\\\n\\midrule\n"
    
    # Sort by groups and then by impact
    def get_sort_key(item):
        metrics = item[1]['metrics']
        return (item[1]['group'], 
                -abs(np.mean([metrics[m] for m in ['BLEU', 'ROUGE-1', 'MC1', 'MC2']])))
    
    current_group = None
    for category, data in sorted(final_results.items(), key=get_sort_key):
        if data['group'] != current_group:
            if current_group is not None:
                latex_table += "\\midrule\n"
            current_group = data['group']
            
        metrics = data['metrics']
        latex_table += f"{data['group'] if data['group'] != current_group else ''} & {category} & "
        latex_table += f"{metrics['BLEU']:.1f} & {metrics['ROUGE-1']:.1f} & "
        latex_table += f"{metrics['MC1']:.1f} & {metrics['MC2']:.1f} & "
        latex_table += f"{data['count']} \\\\\n"
    
    latex_table += "\\bottomrule\n\\end{tabular}\n"
    latex_table += """\\caption{Average percentage change in performance metrics by question category after debiasing,
grouped by category type and averaged across all models and methods. Positive numbers indicate improvement, negative numbers indicate degradation. N indicates number of questions in each category.}\n"""
    latex_table += "\\label{tab:truthfulqa-category-metrics}\n\\end{table}"
    
    # Save outputs
    with open('truthfulqa_category_metrics.tex', 'w') as f:
        f.write(latex_table)
    
    # Save raw data
    raw_data = {category: {
        **{f"{metric}_change": data['metrics'][metric] for metric in data['metrics']},
        'group': data['group'],
        'sample_count': data['count']
    } for category, data in final_results.items()}
    
    pd.DataFrame(raw_data).transpose().to_csv('truthfulqa_category_metrics_raw.csv')
    
    print("Analysis complete! Files saved: truthfulqa_category_metrics.tex and truthfulqa_category_metrics_raw.csv")

if __name__ == "__main__":
    base_dir = 'factual-bias-mitigation/scripts/TruthfulQA/output'
    analyze_category_performance(base_dir)