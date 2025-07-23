# download_model.py
import os
import gdown

MODEL_PATH = "model/densenet.keras"

if not os.path.exists("model"):
    os.makedirs("model")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(id="1oV86Nz4TVGHr83yT2NWBCNeRQdFRiYLa", output=MODEL_PATH, quiet=False)
else:
    print("Model already exists.")
