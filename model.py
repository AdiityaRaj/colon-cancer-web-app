from keras.models import load_model, save_model

# Load your original h5 model
model = load_model("model/densenet.h5", compile=False)

# Save in new .keras format
save_model(model, "model/densenet.keras", save_format="keras")

# Optional: Save as TensorFlow SavedModel format
# model.save("model/saved_model_format")
