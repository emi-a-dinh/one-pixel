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


def multi_pixel_attack(image, num_pixels=10, max_iter=100):
    """Performs an adversarial attack by modifying up to `num_pixels` pixels using any RGB color."""

    def perturbation(params):
        """Applies the pixel modifications and queries the model."""
        img_copy = image.copy()

        for i in range(num_pixels):
            x = int(params[i * 5])
            y = int(params[i * 5 + 1])
            r = int(params[i * 5 + 2])
            g = int(params[i * 5 + 3])
            b = int(params[i * 5 + 4])
            img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        response = call_model(img_copy)  # Call the model API
        return -response["predictions"][0]["probability"]  # Minimize original class probability

    # Define search space: (x, y, r, g, b) for each pixel
    bounds = []
    for _ in range(num_pixels):
        bounds.extend([(0, 64), (0, 64), (0, 255), (0, 255), (0, 255)])  # Any color in RGB space

    # Run optimization
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

    # Apply the optimal perturbation
    adversarial_image = image.copy()
    for i in range(num_pixels):
        x = int(result.x[i * 5])
        y = int(result.x[i * 5 + 1])
        r = int(result.x[i * 5 + 2])
        g = int(result.x[i * 5 + 3])
        b = int(result.x[i * 5 + 4])
        adversarial_image[y, x] = [b, g, r]  # OpenCV uses BGR

    return adversarial_image  # Return the modified NumPy array


def fgsm_attack(image, target_label=1, epsilon=0.1):
    """Performs FGSM attack to push prediction towards `target_label`."""
    img_tensor = tf.convert_to_tensor(np.expand_dims(image / 255.0, axis=0), dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        prediction = model(img_tensor)  # Get model output
        target = tf.convert_to_tensor([[target_label]], dtype=tf.float32)  # Target label tensor
        loss = tf.keras.losses.binary_crossentropy(target, prediction)  # Maximize probability of target_label
    
    gradient = tape.gradient(loss, img_tensor)  # Compute gradients
    signed_grad = tf.sign(gradient)  # Get the sign of the gradient

    adversarial_image = img_tensor + epsilon * signed_grad  # Apply perturbation
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1) * 255  # Keep pixel values valid

    return adversarial_image.numpy().squeeze().astype(np.uint8)




def produce_altered_image(image, pixel):
    altered_image = image.copy()
    x, y, r, g, b = map(int, pixel)
    altered_image[y, x] = [r, g, b]

    return altered_image


path = sys.argv[1]
image = imread(path)
original = call_model(image)
print("Original Prediction:", original)



multi = multi_pixel_attack(image)
print("Multiple attack", call_model(multi))

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

