import os
from huggingface_hub import HfApi
from tqdm import tqdm

# Initialize the Hugging Face API
api = HfApi()

# Set your Hugging Face username and repository name
username = "buseskorkmaz"
repo_name = "factual-bias-mitigation-models"

# Path to the directory containing the models
base_dir = "/gpfs/home/bsk18/factual-bias-mitigation/outputs"

# List of model file extensions to upload
model_extensions = [".pt", ".ckpt", ".bin", ".model", ".pth"]

def should_upload(filename):
    return any(filename.endswith(ext) for ext in model_extensions)

def upload_file(file_path, repo_path):
    try:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=repo_path,
            repo_id=f"{username}/{repo_name}",
            repo_type="model",
        )
        print(f"Uploaded: {repo_path}")
    except Exception as e:
        print(f"Error uploading {file_path}: {str(e)}")

# Get all model files
model_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if should_upload(file):
            model_files.append(os.path.join(root, file))

# Upload each model file
for model_path in tqdm(model_files, desc="Uploading models"):
    try:
        # Get the relative path to use as the filename in the repo
        repo_path = os.path.relpath(model_path, base_dir)
        
        # Upload the file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=repo_path,
            repo_id=f"{username}/{repo_name}",
            repo_type="model",
        )
        print(f"Uploaded: {repo_path}")
    except Exception as e:
        print(f"Error uploading {model_path}: {str(e)}")

print("Upload complete!")

specific_model_path = "/dccstor/nsllm/research/models/factuality/token_type_512_synthetic/model_mnli_snli"


# Now, let's upload all files from the specific model directory
print(f"Uploading files from specific model directory: {specific_model_path}")

if os.path.exists(specific_model_path):
    for root, dirs, files in os.walk(specific_model_path):
        for file in tqdm(files, desc="Uploading specific model files"):
            file_path = os.path.join(root, file)
            repo_path = os.path.join("factuality_detector", os.path.relpath(file_path, specific_model_path))
            upload_file(file_path, repo_path)
else:
    print(f"Specific model directory not found: {specific_model_path}")

print("All uploads complete!")