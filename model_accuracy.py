import os
import random
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize
import cv2
import sys

# Path to your .h5 model
MODEL_PATH = "0.29452_f1max_0.14705_f1_0.78622_loss_0_epoch_model.h5"

# Directory containing images (JPG, PNG, or other formats)
IMAGE_DIR = "../deep-histopath/data/mitoses/patches/val/normal"

# Load the Keras model from .h5
model = load_model(MODEL_PATH)

# Define image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    """Loads an image, resizes it, and preprocesses it for the model."""
    image = cv2.imread(image_path)  # Load the image using OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = resize(image, target_size)  # Resize to match model input
    image = img_to_array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension

def load_random_images(directory, num_samples=1000):
    """Select 100 random image file paths from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Runs an image through the model and returns predictions."""
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return prediction.flatten()  # Convert output to 1D NumPy array

def main():

    image_paths = load_random_images(IMAGE_DIR)
    results = []

    for image_path in image_paths:
        output = run_model_on_image(image_path)
        results.append(output)

    results = np.array(results)  # Convert list to NumPy array for statistics
    mean_values = np.mean(results, axis=0)
    std_values = np.std(results, axis=0)

    print(f"Mean of model outputs:\n {mean_values}")
    print(f"Standard deviation of model outputs:\n {std_values}")

if __name__ == "__main__":
    main()
