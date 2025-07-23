from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import gdown

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "model"
MODEL_PATH = os.path.join(MODEL_FOLDER, "densenet.keras")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    file_id = "1oV86Nz4TVGHr83yT2NWBCNeRQdFRiYLa"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Class names for prediction
class_names = [
    'dyed-lifted-polyps',
    'dyed-resection-margins',
    'esophagitis',
    'normal-cecum',
    'normal-pylorus',
    'normal-z-line',
    'polyps',
    'ulcerative-colitis'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            image_path = file_path

            # Preprocess the uploaded image
            img = load_img(file_path, target_size=(75, 100))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Run prediction
            preds = model.predict(img_array)
            prediction = class_names[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)

    return render_template("index.html", prediction=prediction, confidence=confidence, image=image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)








