from PIL import Image
import numpy as np
import cv2

def preprocess_image(image_file):
    """
    Loads the image file, resizes it to 28x28 pixels, 
    converts it to grayscale, and normalizes pixel values.
    """
    print(f"Loading image: {image_file}")
    img = Image.open(image_file).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    print("Image loaded and resized to 28x28 pixels.")

    img_array = np.array(img)
    thresh = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    img = Image.fromarray(thresh)

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_data = np.expand_dims(img_array, axis=0)
    
    print("Pixel values normalized.")

    return img_data