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



def get_important_pixels(image, num_pixels=1):
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

    # Find the `num_pixels` highest gradient pixels
    indices = np.dstack(np.unravel_index(np.argsort(saliency_map.ravel())[-num_pixels:], saliency_map.shape))
    
    return indices.squeeze()



def targeted_one_pixel(image, important_pixel, max_iter=300):
    """Performs an adversarial attack modifying only one highly important pixel from the saliency map."""
    
    if important_pixel is None or len(important_pixel) == 0:
        raise ValueError("You must provide an important pixel from a saliency map!")

    x, y = important_pixel[0]  # Get the most important pixel

    def perturbation(params):
        """Applies the pixel modification and queries the model."""
        img_copy = image.copy()
        r = int(params[0])
        g = int(params[1])
        b = int(params[2])
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        response = call_model(img_copy)  # Call the model API
        return -response["predictions"][0]["probability"]  # Minimize original class probability

    # Bounds for RGB values only (since x, y are preselected)
    bounds = [(0, 255), (0, 255), (0, 255)]

    # Run optimization
    result = differential_evolution(perturbation, bounds, maxiter=max_iter, strategy='best1bin', popsize=20)

    # Apply the optimal perturbation
    adversarial_image = image.copy()
    r = int(result.x[0])
    g = int(result.x[1])
    b = int(result.x[2])
    adversarial_image[y, x] = [b, g, r]  # OpenCV uses BGR format

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


important_pixels = get_important_pixels(image, num_pixels=1)
adversarial_image = targeted_one_pixel(image, important_pixels)

print("Multiple attack", call_model(adversarial_image))

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

