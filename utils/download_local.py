# ── Local Kaggle Dataset Downloader ──────────────────────────────────────────
# Run this script on your local Windows computer to download the dataset.
# Since the BOSCH server proxy blocks 'api.kaggle.com', downloading locally is 
# the easiest way to fetch the dataset.

import os
import sys
import ssl
import subprocess
import shutil
from pathlib import Path
import warnings

# Bypass SSL Verification for urllib (standard urllib requests)
ssl._create_default_https_context = ssl._create_unverified_context

# Bypass SSL Verification for requests (third-party libraries like kaggle-api)
import requests
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# Store the original method
old_merge_environment_settings = requests.Session.merge_environment_settings

# Define the monkeypatch
def merge_environment_settings(self, url, proxies, stream, verify, cert):
    settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
    settings['verify'] = False
    return settings

# Apply the patch
requests.Session.merge_environment_settings = merge_environment_settings

DATASET_SLUG = "thnhdg/testing"
KAGGLE_USERNAME = "nctuan"
KAGGLE_KEY = "5684571d94f144c2d35e31f9c96dc5f1"

def install_kaggle_local():
    try:
        import kaggle
        print("[INFO] Kaggle package is already installed locally.")
    except ImportError:
        print("[INFO] Kaggle package not found. Installing locally...")
        # Disable SSL warning during pip install or use standard pip
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)

def setup_credentials():
    os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
    os.environ['KAGGLE_KEY'] = KAGGLE_KEY

    # Write to local .kaggle folder
    home_dir = Path.home()
    kaggle_dir = home_dir / '.kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    config_path = kaggle_dir / 'kaggle.json'
    
    import json
    with open(config_path, 'w') as f:
        json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)
    
    try:
        os.chmod(config_path, 0o600)
    except Exception:
        pass  # ignore permission errors on Windows
    print(f"[INFO] Kaggle credentials configured locally at: {config_path}")

def download_dataset():
    # Target path inside the local repository
    script_dir = Path(__file__).parent
    dest_dir = script_dir / "data" / "phase1"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Downloading dataset '{DATASET_SLUG}' to '{dest_dir.resolve()}'...")
    
    # Try Option A: Download via Python API (runs with the main thread SSL bypass)
    success = False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Download files to destination
        api.dataset_download_files(DATASET_SLUG, path=str(dest_dir), unzip=False, quiet=False)
        print("[SUCCESS] Dataset downloaded successfully via Python API!")
        success = True
    except Exception as e:
        print(f"[WARNING] Python API download failed: {e}")
        print("[INFO] Trying fallback to Kaggle CLI...")
        
        # Try Option B: Call CLI via python entrypoint with SSL bypass passed to subprocess
        cli_code = (
            "import ssl, requests, warnings; "
            "from urllib3.exceptions import InsecureRequestWarning; "
            "warnings.simplefilter('ignore', InsecureRequestWarning); "
            "ssl._create_default_https_context = ssl._create_unverified_context; "
            "old_merge = requests.Session.merge_environment_settings; "
            "requests.Session.merge_environment_settings = lambda self, url, proxies, stream, verify, cert: "
            "{**old_merge(self, url, proxies, stream, verify, cert), 'verify': False}; "
            "from kaggle.cli import main; main()"
        )
        
        cli_cmd = [
            sys.executable, "-c", cli_code,
            "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(dest_dir)
        ]
        
        result = subprocess.run(cli_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("[SUCCESS] Dataset downloaded successfully via CLI!")
            success = True
        else:
            print("[ERROR] Kaggle CLI download failed:")
            print(result.stderr)
            print(result.stdout)
            
    if success:
        # Verify the file was downloaded (sometimes it downloads as testing.zip)
        expected_zip = dest_dir / "testing.zip"
        if expected_zip.exists():
            print(f"[SUCCESS] File saved at: {expected_zip.resolve()}")
            print("\nNext Steps:")
            print("1. Open JupyterLab on your remote server.")
            print("2. Navigate to your target directory: /home/ghp4hc/datasets/datasets/mipneft360/")
            print("3. Use the 'Upload' button (upward arrow ↑ in the left sidebar) to upload 'testing.zip'.")
            print("4. Run the last cell in the notebook to extract it.")
        else:
            # Check what files were downloaded
            downloaded = list(dest_dir.glob("*.zip"))
            if downloaded:
                print(f"[INFO] Found downloaded file: {downloaded[0]}")
                print("\nNext Steps:")
                print("1. Open JupyterLab on your remote server.")
                print("2. Navigate to your target directory: /home/ghp4hc/datasets/datasets/mipneft360/")
                print(f"3. Use the 'Upload' button to upload '{downloaded[0].name}'.")
                print("4. Run the last cell in the notebook (update the zip filename if it is not 'testing.zip') to extract it.")
            else:
                print("[ERROR] Download finished but zip file not found in destination directory.")

if __name__ == "__main__":
    try:
        install_kaggle_local()
        setup_credentials()
        download_dataset()
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        input("\nPress Enter to exit...")
