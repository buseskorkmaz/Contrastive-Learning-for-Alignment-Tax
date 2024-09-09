import json
import sys
from statistics import mean

def process_seat_results(file_path):
    # Read the JSONL file
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file][0]

    # Extract model name from the first experiment_id
    model_name = data[0]['experiment_id'].split('_')[2]

    # Categorize tests
    categories = {
        'SEAT-6': [], 'SEAT-6b': [],
        'SEAT-7': [], 'SEAT-7b': [],
        'SEAT-8': [], 'SEAT-8b': []
    }

    for item in data:
        test_name = item['test']
        if test_name.startswith('weat') or test_name.startswith('sent-weat'):
            number = test_name.split('weat')[-1]
            if number in ['6', '7', '8']:
                category = f'SEAT-{number}'
                categories[category].append(item['effect_size'])
            elif number in ['6b', '7b', '8b']:
                category = f'SEAT-{number}'
                categories[category].append(item['effect_size'])

    # Calculate average effect sizes for each category
    category_averages = {cat: mean(sizes) if sizes else 0 for cat, sizes in categories.items()}

    # Calculate overall average absolute effect size
    all_effects = [abs(effect) for effects in categories.values() for effect in effects]
    overall_avg = mean(all_effects) if all_effects else 0

    # Print summary
    print(f"SEAT Results Summary for {model_name}")
    print("\nEffect sizes by category:")
    for cat, avg in category_averages.items():
        print(f"  {cat}: {avg:.3f}")
    print(f"\nOverall average absolute effect size: {overall_avg:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_jsonl_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    process_seat_results(file_path)