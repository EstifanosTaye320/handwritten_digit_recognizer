from tensorflow.keras.models import load_model
import numpy as np

def load_mnist_model():
    """Loads the trained MNIST model from a saved file."""
    try:
        model = load_model("mnist_model.h5")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def predict_digit(model, image_data):
    """Predicts the digit using the provided model and image data."""
    prediction = model.predict(image_data)
    digit = np.argmax(prediction)

    return digit, prediction