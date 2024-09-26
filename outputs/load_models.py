import os
from huggingface_hub import HfApi
from tqdm import tqdm

# Initialize the Hugging Face API
api = HfApi()

# Set your Hugging Face username and repository name
username = "buseskorkmaz"
repo_name = "factual-bias-mitigation-models"

# Path to the directory containing the models
base_dir = "/dccstor/autofair/busekorkmaz/factual-bias-mitigation/outputs"

# List of model file extensions to upload
model_extensions = [".pt", ".ckpt", ".bin", ".model", ".pth"]

def should_upload(filename):
    return any(filename.endswith(ext) for ext in model_extensions)

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