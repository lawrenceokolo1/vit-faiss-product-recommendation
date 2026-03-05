"""Upload space/ folder to Hugging Face Space. Reads HF_TOKEN and HF_SPACE_REPO_ID from env."""
import os
from pathlib import Path

from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_SPACE_REPO_ID")
    if not token or not repo_id:
        raise SystemExit("HF_TOKEN and HF_SPACE_REPO_ID must be set")
    folder = Path(__file__).resolve().parents[2] / "space"
    if not folder.is_dir():
        raise SystemExit("space/ folder not found")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="space",
        path_in_repo=".",
    )
    print(f"Deployed to https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    main()
