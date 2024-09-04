import pandas as pd


model_names = ['inlp', 'sentence_debiasing', 'gpt2-tldr', 'instructive_debiasing', 'self_debiasing']
output_files = {
    model_name: f"/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/{model_name}_results.csv" for model_name in model_names
    }

results = {}

for model_name in model_names:
    results[model_name] = pd.read_csv(output_files[model_name])
    avg_faithfulness = results[model_name]['faithfulness'].mean()
    std_faithfulness = results[model_name]['faithfulness'].std()
    avg_generated_toxicity = results[model_name]['generated_toxicity'].mean()
    std_generated_toxicity = results[model_name]['generated_toxicity'].std()
    avg_original_toxicity = results[model_name]['original_toxicity'].mean()
    std_original_toxicity = results[model_name]['original_toxicity'].std()
    print(f"Faithfulness of {model_name}:{avg_faithfulness} +- {std_faithfulness}")
    print(f"Original toxicity of {model_name}:{avg_original_toxicity} +- {std_original_toxicity}")
    print(f"Generated toxicity of {model_name}:{avg_generated_toxicity} +- {std_generated_toxicity}")
    print("--"*50)
    print("\n")