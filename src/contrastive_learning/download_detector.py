from huggingface_hub import snapshot_download
import os
import shutil

def download_subfolder(repo_id, subfolder, local_dir):
    """
    Download a specific subfolder from a Hugging Face repository.
    
    :param repo_id: The repository ID (e.g., 'username/repo-name')
    :param subfolder: The subfolder path within the repository
    :param local_dir: The local directory to save the downloaded files
    """
    # Download the entire repository (or the subfolder if possible)
    repo_dir = snapshot_download(repo_id=repo_id, ignore_patterns="*.pt")
    
    # Construct the path to the subfolder within the downloaded repository
    subfolder_path = os.path.join(repo_dir, subfolder)
    
    # Check if the subfolder exists
    if not os.path.exists(subfolder_path):
        raise ValueError(f"Subfolder '{subfolder}' not found in the repository.")
    
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Copy the contents of the subfolder to the local directory
    for item in os.listdir(subfolder_path):
        s = os.path.join(subfolder_path, item)
        d = os.path.join(local_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    
    print(f"Downloaded subfolder '{subfolder}' to '{local_dir}'")
    
    # Optionally, remove the temporary downloaded repository
    shutil.rmtree(repo_dir)

# Example usage
repo_id = "buseskorkmaz/factual-bias-mitigation-models"
subfolder = "factuality_detector"
local_dir = "./downloaded_files"

download_subfolder(repo_id, subfolder, local_dir)