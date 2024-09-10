import pandas as pd
import numpy as np

def format_value(value, std):
    return f"{value:.3f} $\\pm$ {std:.3f}"

def format_change(base_value, new_value, metric):
    change = new_value - base_value
    if abs(change) < 1e-6:  # To handle floating point precision issues
        return "0.000"
    elif change > 0:
        color_command = r"\increasered" if metric == "toxicity" else r"\increase"
        return f"{color_command}{{{change:.3f}}}"
    else:
        color_command = r"\decreasegreen" if metric == "toxicity" else r"\decrease"
        return f"{color_command}{{{abs(change):.3f}}}"

def generate_latex_table():
    model_families = {
        'gpt2': ['gpt2', 'gpt2_cda_gender', 'gpt2_cda_race', 'gpt2_cda_religion', 'gpt2_dropout',
                 'inlp-gender', 'inlp-race', 'sentence_debiasing-gender', 'sentence_debiasing-race',
                 'self_debiasing-gpt2', 'instructive_debiasing-gpt2'],
        'phi2': ['phi2', 'phi2_cda_gender', 'phi2_cda_religion', 'phi2_cda_race', 'phi2_dropout','self_debiasing-phi2',
                 'instructive_debiasing-phi2'],
        'llama2': ['llama2-7b', 'llama2-7b_cda_gender', 'llama2-7b_cda_religion', 'llama2-7b_cda_race',
                   'llama2-7b_dropout','self_debiasing-llama2', 'instructive_debiasing-llama2']
    }

    output_files = {
        model_name: f"/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/{model_name}_results.csv"
        for family in model_families.values() for model_name in family
    }

    latex_table = r"""
\begin{table}[h]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\multirow{2}{*}{Model} & \multicolumn{2}{c}{Original Toxicity} & \multicolumn{2}{c}{Generated Toxicity} & \multicolumn{2}{c}{Faithfulness} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
& Value & Change & Value & Change & Value & Change \\
\midrule
"""

    for family, models in model_families.items():
        base_model = models[0]
        base_model_data = pd.read_csv(output_files[base_model])
        base_original_toxicity = base_model_data['original_toxicity'].mean()
        base_generated_toxicity = base_model_data['generated_toxicity'].mean()
        base_faithfulness = base_model_data['faithfulness'].mean()

        for model_name in models:
            df = pd.read_csv(output_files[model_name])
            
            avg_faithfulness = df['faithfulness'].mean()
            std_faithfulness = df['faithfulness'].std()
            avg_generated_toxicity = df['generated_toxicity'].mean()
            std_generated_toxicity = df['generated_toxicity'].std()
            avg_original_toxicity = df['original_toxicity'].mean()
            std_original_toxicity = df['original_toxicity'].std()

            model_name_formatted = model_name.replace('_', ' ').title()
            
            if model_name == base_model:
                latex_table += f"{model_name_formatted} & {format_value(avg_original_toxicity, std_original_toxicity)} & - & {format_value(avg_generated_toxicity, std_generated_toxicity)} & - & {format_value(avg_faithfulness, std_faithfulness)} & - \\\\\n"
            else:
                latex_table += f"{model_name_formatted} & {format_value(avg_original_toxicity, std_original_toxicity)} & {format_change(base_original_toxicity, avg_original_toxicity, 'toxicity')} & {format_value(avg_generated_toxicity, std_generated_toxicity)} & {format_change(base_generated_toxicity, avg_generated_toxicity, 'toxicity')} & {format_value(avg_faithfulness, std_faithfulness)} & {format_change(base_faithfulness, avg_faithfulness, 'faithfulness')} \\\\\n"
        
        if family != list(model_families.keys())[-1]:
            latex_table += r"\midrule" + "\n"

    latex_table += r"""\bottomrule
\end{tabular}
\caption{Model Performance: Toxicity and Faithfulness (compared to respective base models)}
\label{tab:model-performance-comparative}
\end{table}
"""
    return latex_table

# Generate and print the LaTeX table
print(generate_latex_table())