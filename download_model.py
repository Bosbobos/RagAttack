from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
filename = "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
local_dir = "models"

if __name__ == "__main__":
    os.makedirs(local_dir, exist_ok=True)

    file_path = hf_hub_download(repo_id=repo_id,
                                filename=filename,
                                cache_dir=repo_id)

    shutil.move(file_path, local_dir)
    os.remove(repo_id)
