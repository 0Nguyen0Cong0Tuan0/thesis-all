# ── Upload Dataset to Hugging Face ─────────────────────────────────────────────
# Run this script on your local Windows PC once the Kaggle download completes.
# It uses your token from 'hf_token.json' to create a private Hugging Face dataset
# and upload 'testing.zip'.

import os
import json
import ssl
from pathlib import Path

# 1. Load Hugging Face token
token_path = Path(__file__).parent / "hf_token.json"
if not token_path.exists():
    print(f"[ERROR] hf_token.json not found at {token_path.resolve()}. Please make sure it is in the repository root.")
    exit(1)

with open(token_path, "r") as f:
    token_data = json.load(f)
HF_TOKEN = token_data.get("HF_KEY")

if not HF_TOKEN:
    print("[ERROR] HF_KEY not found in hf_token.json.")
    exit(1)

# Dataset details
DATASET_REPO = os.environ.get("HF_REPO", "DiBiay/testing-dataset")

zip_candidates = [
    Path(__file__).parent / "drive-download-20260716T045906Z-1-001.zip",
    Path(__file__).parent / "data" / "phase1" / "testing.zip",
    Path(__file__).parent / "testing.zip",
]

ZIP_PATH = next((p for p in zip_candidates if p.exists()), None)

if not ZIP_PATH:
    print(f"\n[ERROR] No candidate zip file found in paths: {[str(p.resolve()) for p in zip_candidates]}")
    print("Please make sure 'drive-download-20260716T045906Z-1-001.zip' or 'testing.zip' is in the workspace folder!")
    exit(1)

print(f"Using local zip file: {ZIP_PATH.resolve()} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")

# 2. Check and install huggingface_hub if missing
print("Verifying huggingface_hub library...")
try:
    import huggingface_hub
except ImportError:
    print("huggingface_hub not found. Installing locally...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
    import huggingface_hub

from huggingface_hub import HfApi

# 3. Configure SSL verification bypass (for corporate proxy compatibility)
ssl._create_default_https_context = ssl._create_unverified_context
import requests
from urllib3.exceptions import InsecureRequestWarning
import warnings
warnings.simplefilter('ignore', InsecureRequestWarning)

old_merge = requests.Session.merge_environment_settings
requests.Session.merge_environment_settings = lambda self, url, proxies, stream, verify, cert: \
    {**old_merge(self, url, proxies, stream, verify, cert), 'verify': False}

# 4. Initialize API and create repository
api = HfApi(token=HF_TOKEN)

print(f"\nCreating private Hugging Face dataset repository '{DATASET_REPO}'...")
try:
    api.create_repo(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True
    )
    print("[SUCCESS] Dataset repository verified.")
except Exception as e:
    print(f"[ERROR] Failed to create repository: {e}")
    exit(1)

# 5. Upload file
print(f"\nUploading {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB) to Hugging Face...")
print("This may take some time depending on your upload speed. Please do not close this window...")
try:
    api.upload_file(
        path_or_fileobj=str(ZIP_PATH),
        path_in_repo=ZIP_PATH.name,
        repo_id=DATASET_REPO,
        repo_type="dataset"
    )
    print("\n[SUCCESS] Upload complete!")
    print(f"Dataset is now available privately on Hugging Face: https://huggingface.co/datasets/{DATASET_REPO}")
    print("\nNext Steps:")
    print("1. Sync the repository or open JupyterLab on the remote server.")
    print("2. Run the code block in the notebook to download from Hugging Face.")
except Exception as e:
    print(f"\n[ERROR] Upload failed: {e}")
