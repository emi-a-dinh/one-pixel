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


def call_model(image_array):
    """Runs the local HDF5 model on the input image."""
    img_pil = Image.fromarray(np.uint8(image_array))

    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))

    img_np = np.array(img_pil) / 255.0  # Normalize pixel values
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

    predictions = model.predict(img_np)

    # Apply correct activation function based on model type
    if predictions.shape[-1] == 1:  # Binary classification (single output neuron)
        probability = tf.nn.sigmoid(predictions).numpy()[0][0]
    else:  # Multi-class classification (more than 1 output neuron)
        probability = tf.nn.softmax(predictions).numpy()[0].tolist()

    return probability

def load_random_images(directory, num_samples=100):
    """Selects 100 random images from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Loads an image and runs it through the model."""
    image = Image.open(image_path).convert("RGB")  # Open as RGB
    image = np.array(image)  # Convert to NumPy array
    return call_model(image)  # Get predictions

def main():
    image_paths = load_random_images(IMAGE_DIR)
    results = []

    for image_path in image_paths:
        output = run_model_on_image(image_path)
        results.append(output)

    results = np.array(results)  # Convert list to NumPy array
    mean_values = np.mean(results, axis=0)
    std_values = np.std(results, axis=0)

    print(f"Mean of model outputs:\n {mean_values}")
    print(f"Standard deviation of model outputs:\n {std_values}")

if __name__ == "__main__":
    main()