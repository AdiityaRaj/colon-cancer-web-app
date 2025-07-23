import os
import gdown

MODEL_DIR = "model"
MODEL_FILENAME = "densenet.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
FILE_ID = "1oV86Nz4TVGHr83yT2NWBCNeRQdFRiYLa"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
else:
    print("✅ Model already exists.")

