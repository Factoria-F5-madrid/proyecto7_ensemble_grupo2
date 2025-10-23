# download_from_drive.py
import os
import zipfile
import gdown

DRIVE_URL = os.environ.get("DRIVE_URL") or "https://drive.google.com/uc?id=1ReMjp7GNUz1cm2o-V8nDDdtAO9shnUtE"
ZIP_PATH = "data/desayuno_preprocessed_archive.zip"
DATASET_DIR = "data/desayuno_preprocessed"

os.makedirs("data", exist_ok=True)

print("Downloading ZIP from Google Drive...")
gdown.download(DRIVE_URL, ZIP_PATH, quiet=False)

if os.path.getsize(ZIP_PATH) < 1000000:
    raise ValueError("Downloaded file too small; check DRIVE_URL")

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)
print("Extracted to:", DATASET_DIR)
