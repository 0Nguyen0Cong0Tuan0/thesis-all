# ── Local Google Drive Downloader for datar2 ─────────────────────────────────
# Downloads dataset from Google Drive (ID: 1b9F4B1tDVX8bIX4fZxsP9bduRynDUN_a),
# saves it as 'datar2.zip', and unzips it into the 'datar2' directory.

import os
import sys
import ssl
import zipfile
import subprocess
from pathlib import Path
import warnings

# Ensure UTF-8 output encoding for console stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Bypass SSL verification for urllib & requests (useful for corporate networks)
ssl._create_default_https_context = ssl._create_unverified_context
import requests
from urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

old_merge = requests.Session.merge_environment_settings
requests.Session.merge_environment_settings = lambda self, url, proxies, stream, verify, cert: \
    {**old_merge(self, url, proxies, stream, verify, cert), 'verify': False}

# 2. Ensure gdown is installed
try:
    import gdown
except ImportError:
    print("gdown not found. Installing gdown...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
    import gdown

# 3. Google Drive file configuration
FILE_ID = "1b9F4B1tDVX8bIX4fZxsP9bduRynDUN_a"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

BASE_DIR = Path(__file__).parent
ZIP_PATH = BASE_DIR / "datar2.zip"
EXTRACT_DIR = BASE_DIR / "datar2"

print(f"[INFO] Starting download from Google Drive (ID: {FILE_ID}) ...")
print(f"[INFO] Target zip path : {ZIP_PATH.resolve()}")

# Download using gdown with direct ID URL
output = gdown.download(URL, str(ZIP_PATH), quiet=False)

if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size == 0:
    print(f"\n[ERROR] Download failed or file is empty: {ZIP_PATH}")
    sys.exit(1)

size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"\n[SUCCESS] Download complete! Saved to '{ZIP_PATH.name}' ({size_mb:.2f} MB)")

# 4. Unzip dataset into datar2 directory
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n[INFO] Extracting '{ZIP_PATH.name}' into '{EXTRACT_DIR.resolve()}'...")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print(f"[SUCCESS] Dataset successfully extracted to: {EXTRACT_DIR.resolve()}")

# Verify extracted contents
print("\n[SUMMARY] Extracted contents summary:")
files = sorted(os.listdir(EXTRACT_DIR))
print(f"Total items in {EXTRACT_DIR.name}: {len(files)}")
for f in files[:20]:
    item_p = EXTRACT_DIR / f
    if item_p.is_dir():
        print(f" [DIR]  {f}/")
    else:
        print(f" [FILE] {f} ({item_p.stat().st_size / (1024 * 1024):.2f} MB)")
