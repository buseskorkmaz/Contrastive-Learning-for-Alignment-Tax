import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Your existing code to load and process data
models = ['original', 'gpt2', 'gpt2_cda_gender', 'gpt2_cda_race', 'gpt2_cda_religion', 'gpt2_dropout',
          'inlp-gender', 'inlp-race', 'sentence_debiasing-gender', 'sentence_debiasing-race',
          'self_debiasing-gpt2', 'instructive_debiasing-gpt2', 'phi2', 'phi2_cda_gender', 'phi2_cda_religion', 'phi2_cda_race', 'phi2_dropout','self_debiasing-phi2',
          'instructive_debiasing-phi2','llama2-7b', 'llama2-7b_cda_gender', 'llama2-7b_cda_religion', 'llama2-7b_cda_race',
          'llama2-7b_dropout','self_debiasing-llama2', 'instructive_debiasing-llama2']
files = [f"/dccstor/autofair/busekorkmaz/factual-bias-mitigation/scripts/summarization/output/{model_name}_results.csv" for model_name in models]

df_list = (pd.read_csv(file) for file in files)
df = pd.concat(df_list, ignore_index=True)
df_dedup = df.drop_duplicates(subset=['generated_summary'])
pos_data = df_dedup[(df_dedup.faithfulness >= 0.9) & (df_dedup.generated_toxicity < 0.159)]
neg_data = df_dedup[(df_dedup.faithfulness < 0.9) & (df_dedup.generated_toxicity >= 0.159)]

# Updated function to process and save data
def process_and_save_data(data, data_type):
    # Perform train/test/validation split
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    # Define directories
    base_dir = "/dccstor/autofair/busekorkmaz/factual-bias-mitigation/data/tldr/" + data_type
    
    # Function to save data
    def save_split(split_data, split_name):
        # Prepare source data
        split_data['source'] = split_data['content'].str.replace('\n', ' ')
        split_data['generated_summary'] = split_data['generated_summary'].str.replace('\n', ' ')
        # Save source data
        with open(os.path.join(base_dir, f"{split_name}.source"), 'w', encoding='utf-8') as f:
            for line in split_data['source']:
                f.write(line.strip() + '\n')
        
        # Save target data
        with open(os.path.join(base_dir, f"{split_name}.target"), 'w', encoding='utf-8') as f:
            for line in split_data['generated_summary']:
                f.write(line.strip() + '\n')
    
    # Save data for each split
    save_split(train_data, 'train')
    save_split(test_data, 'test')
    save_split(val_data, 'validation')

# Process and save positive and negative data
process_and_save_data(pos_data, 'pos')
process_and_save_data(neg_data, 'neg')

print("Data processing and saving completed.")