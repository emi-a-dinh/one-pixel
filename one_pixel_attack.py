import numpy as np
import h5py
import tensorflow as tf
from cv2 import imread, imwrite
from scipy.optimize import differential_evolution
from PIL import Image
from tensorflow.keras.models import model_from_json
import os
import requests
import sys
import cv2
import tempfile
import io
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

#upgraded tensor flow from 1.15.4 to 2.07

MODEL_PATH = "0.29452_f1max_0.14705_f1_0.78622_loss_0_epoch_model.h5"  # Path to your HDF5 model file

# # Define the model architecture manually (make sure it matches the original one)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Load weights from HDF5
model.load_weights(MODEL_PATH)
model.summary()




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

    return {"predictions": [{"probability": float(probability)}]}



def get_important_pixels(image, num_pixels=10):
    """Finds the top `num_pixels` pixels that contribute the most to the model’s decision."""
    
    # Ensure input is properly formatted
    img_np = np.array(image) / 255.0  # Normalize image
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

    # Define loss function to maximize the predicted class
    def loss_function(output):
        return output[:, 0]  # Use the first class probability

    # Initialize Saliency object with a model modifier
    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=False)

    # Compute saliency map
    saliency_map = saliency(loss_function, img_np)
    saliency_map = np.squeeze(saliency_map)  # Remove batch dimension

    # Find `num_pixels` highest gradient pixels
    indices = np.unravel_index(np.argsort(saliency_map.ravel())[-num_pixels:], saliency_map.shape)

    # Convert indices to (x, y) format
    important_pixels = list(zip(indices[1], indices[0]))  # Convert to (x, y) tuples

    return important_pixels 



def targeted_one_pixel(image, important_pixels, max_iter=300):
    """Performs an adversarial attack modifying only one highly important pixel from the saliency map."""
    
    if important_pixels is None or len(important_pixels) == 0:
        raise ValueError("You must provide at least one important pixel from a saliency map!")

    best_pixel = None
    best_prob = float('-inf')  # Track the best probability change
    best_adversarial_image = None

    for x, y in important_pixels:  # Iterate through all top pixels

        def perturbation(params):
            img_copy = image.copy()
            r = int(params[0])
            g = int(params[1])
            b = int(params[2])
            img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

            response = call_model(img_copy)  # Call the model API
            prob = response["predictions"][0]["probability"]
            
            return -prob  # Minimize the original class probability

        # Bounds for RGB values
        bounds = [(0, 255), (0, 255), (0, 255)]

        # Run optimization
        result = differential_evolution(perturbation, bounds, maxiter=max_iter, strategy='best1bin', popsize=50)

        # Apply the optimal perturbation
        adversarial_image = image.copy()
        r = int(result.x[0])
        g = int(result.x[1])
        b = int(result.x[2])
        adversarial_image[y, x] = [b, g, r]  # OpenCV uses BGR format

        # Get the new prediction
        new_prediction = call_model(adversarial_image)["predictions"][0]["probability"]

        # Keep track of the best pixel
        if new_prediction > best_prob:
            best_prob = new_prediction
            best_pixel = (x, y, r, g, b)
            best_adversarial_image = adversarial_image.copy()

    print(f"Best Pixel: {best_pixel} | Best Probability: {best_prob}")
    return best_adversarial_image  # Return the modified NumPy array





def fgsm_attack(image, target_label=None, epsilon=0.2):
    """Performs a fast gradient sign method (FGSM) attack using backpropagation."""

    # Normalize image to [0,1]
    image = image.astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)  # Add batch dimension

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)  # Forward pass

        # If no target_label is provided, assume untargeted attack (push away from current prediction)
        if target_label is None:
            target_label = tf.round(tf.nn.sigmoid(prediction))  # Get the current predicted class (0 or 1)

        # Ensure target_label is a TensorFlow tensor and reshape it to match logits
        target_label_tensor = tf.convert_to_tensor(target_label, dtype=tf.float32)
        target_label_tensor = tf.reshape(target_label_tensor, prediction.shape)  # Ensure same shape

        # Compute loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_label_tensor, logits=prediction)

    # Compute gradients
    gradient = tape.gradient(loss, img_tensor)

    # Apply the FGSM perturbation
    signed_grad = tf.sign(gradient)  # Get sign of the gradient
    adversarial_image = img_tensor + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1).numpy()[0] * 255  # Convert back to [0,255]

    return adversarial_image.astype(np.uint8)


def produce_altered_image(image, pixel):
    altered_image = image.copy()
    x, y, r, g, b = map(int, pixel)
    altered_image[y, x] = [r, g, b]

    return altered_image


path = sys.argv[1]
image = imread(path)
original = call_model(image)
print("Original Prediction:", original)


important_pixels = get_important_pixels(image, num_pixels=1)
adversarial_image = targeted_one_pixel(image, important_pixels)

/bin/bash: line 1: q: command not found

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

