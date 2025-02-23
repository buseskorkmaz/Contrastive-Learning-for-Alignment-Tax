import json
import os
import glob
import matplotlib.pyplot as plt
from toxicity import ToxicityScorer

def inspect_dataset(dir_path, dataset_type):
    """Helper function to inspect dataset format and content"""
    print(f"\n=== Inspecting {dataset_type} dataset ===")
    try:
        # Load the dataset
        from datasets import load_from_disk
        dataset = load_from_disk(dir_path)
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Features: {dataset.features}")
        print(f"Column names: {dataset.column_names}")
        
        # Look at first few examples
        print("\nFirst 3 examples:")
        for i in range(min(3, len(dataset))):
            print(f"\nExample {i}:")
            example = dataset[i]
            for col in dataset.column_names:
                if isinstance(example[col], (str, bool, int, float)):
                    print(f"{col}: {example[col]}")
                else:
                    print(f"{col}: {type(example[col])}")
        
        # Print some statistics
        print("\nField statistics:")
        for col in dataset.column_names:
            empty_count = sum(
                1 for i in range(len(dataset))
                if isinstance(dataset[i][col], str) and not dataset[i][col]
            )
            print(f"{col} - empty fields: {empty_count}")
            
            # Print a few examples of non-empty values
            print(f"Sample non-empty values for {col}:")
            non_empty_examples = [
                dataset[i][col] 
                for i in range(min(3, len(dataset))) 
                if isinstance(dataset[i][col], str) and dataset[i][col]
            ]
            for example in non_empty_examples[:3]:
                print(f"  - {example[:100]}...")
                
        # Load metadata if exists
        metadata_path = os.path.join(dir_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print("\nMetadata:")
                print(json.dumps(metadata, indent=2))
                
    except Exception as e:
        print(f"Error inspecting dataset: {e}")

# Test with each dataset type
base_dir = '/home/ec2-user/busesk/constrastove_training/data/processed_filled'
directories = {
    'back_translation': os.path.join(base_dir, 'pos/back_translation'),
    'original': os.path.join(base_dir, 'pos/original'),
    'swap_ent': os.path.join(base_dir, 'neg/swap_ent')
}

for dataset_type, dir_path in directories.items():
    inspect_dataset(dir_path, dataset_type)

# Original script continues...
directory = '/home/ec2-user/busesk/constrastove_training/output/runs/2024-11-20/11-40-24/output/cache/scores/'
json_files = glob.glob(os.path.join(directory, '*.json'))

# Initialize cumulative variables
faithfulness_scores = []
toxicity_scores = []
total_toxicity_score = 0
total_samples = 0

# Process each JSON file
for json_file in json_files:
    with open(json_file, 'r') as file:
        scores = json.load(file)
    
    print(f"\nInspecting scores file: {json_file}")
    print(f"Number of samples: {len(scores['samples'])}")
    print(f"Sample structure: {list(scores['samples'][0].keys())}")
    
    # Collect faithfulness scores and accumulate toxicity scores
    for sample in scores['samples']:
        faithfulness_scores.append(sample['faithfulness'])
        toxicity_scores.append(sample['toxicity'])
        total_toxicity_score += sample['toxicity']
    
    total_samples += len(scores['samples'])

# Calculate averages
if total_samples > 0:
    avg_faithfulness = sum(faithfulness_scores) / total_samples
    avg_toxicity = total_toxicity_score / total_samples
else:
    avg_faithfulness = 0
    avg_toxicity = 0

# Print the results with more detail
print("\n=== Overall Statistics ===")
print(f"Total samples processed: {total_samples}")
print(f"Average Faithfulness Score: {avg_faithfulness:.4f}")
print(f"Average Toxicity Score: {avg_toxicity:.4f}")
print(f"Min Faithfulness: {min(faithfulness_scores):.4f}")
print(f"Max Faithfulness: {max(faithfulness_scores):.4f}")
print(f"Min Toxicity: {min(toxicity_scores):.4f}")
print(f"Max Toxicity: {max(toxicity_scores):.4f}")

pos_samples = [sample for sample in scores['samples'] if sample['faithfulness'] >= 0.4 and sample['toxicity'] >= 0.008]
neg_samples = [sample for sample in scores['samples'] if sample['faithfulness'] < 0.4 or sample['toxicity'] > 0.008]

print("\n=== Sample Categorization ===")
print(f"Positive samples: {len(pos_samples)}")
print(f"Negative samples: {len(neg_samples)}")

# Print examples of very low toxicity samples
print("\n=== Low Toxicity Samples ===")
low_tox_samples = [sample for sample in scores['samples'] if sample['toxicity'] < 0.008]
for sample in low_tox_samples[:3]:  # Show first 3 examples
    print(f"Toxicity: {sample['toxicity']:.4f}")
    print(f"Faithfulness: {sample['faithfulness']:.4f}")
    print(f"Post: {sample['post'][:100]}...")
    print(f"Summary: {sample['summary'][:100]}...")
    print("---")

# Plot distributions
plt.figure(figsize=(15, 5))

# Toxicity distribution
plt.subplot(1, 2, 1)
plt.hist(toxicity_scores, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Toxicity Scores')
plt.xlabel('Toxicity Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Faithfulness distribution
plt.subplot(1, 2, 2)
plt.hist(faithfulness_scores, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Faithfulness Scores')
plt.xlabel('Faithfulness Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('distributions.png')

# Test toxicity scorer
scorer = ToxicityScorer(model_name='s-nlp/roberta_toxicity_classifier')
test_texts = [
    "I will fuck and kill you",
    "The whole town knows what sex with DemonOWA sounds like.",
    "thanks, but I already have 7 shit-loads :)",
    "Creepy prof was flirty and kind of hitting on me...",
    "Got drunk while on the clock...",
    "I'm going to need to sue someone.",
    "do the EV Robot dance and quit."
]
toxicity_results = scorer.score_batch(test_texts)
print("\n=== Test Toxicity Scores ===")
for text, score in zip(test_texts, toxicity_results):
    print(f"Score: {score:.4f} | Text: {text[:50]}...")