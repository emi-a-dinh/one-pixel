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


#upgraded tensor flow from 1.15.4 to 2.07

MODEL_PATH = "0.29452_f1max_0.14705_f1_0.78622_loss_0_epoch_model.hdf5"  # Path to your HDF5 model file

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

MODELAPI = "http://0.0.0.0:5000/model/predict"


def call_modelapi(image_path):
    # Ensure the input is a valid file path
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise ValueError("Invalid input: Provide a valid image file path.")

    # Open the image file in binary mode and send it
    with open(image_path, "rb") as img_file:
        files = {"image": (image_path, img_file, "image/png")}
        response = requests.post(MODELAPI, files=files)
    
    return response.json()  # Return the JSON response

def one_pixel_attackapi(image_path, preset_colors, max_iter=100):
    # Load the image
    image = cv2.imread(image_path)  # OpenCV loads in BGR format

    if image is None:
        raise ValueError("Error loading image. Check the file path.")

    height, width, _ = image.shape  # Get image dimensions

    def perturbation(params):
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]

        # Make a copy and apply the perturbation
        img_copy = image.copy()
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        # Save perturbed image to a temporary file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_path = temp_img.name
        temp_img.close()  # Close so OpenCV can write to it
        cv2.imwrite(temp_path, img_copy)

        print(f"Temporary image saved at: {temp_path}")  # Debugging output

        # Check if the file exists before calling API
        if not os.path.isfile(temp_path):
            print(f"Error: Temporary file {temp_path} does not exist!")
            return float("inf")  # Return a high cost if image is not valid

        # Call the model API with the new image
        response = call_modelapi(temp_path)

        # Remove the temporary image file after sending
        os.remove(temp_path)

        return -response["predictions"][0]["probability"]  # Minimize probability

    # Define bounds for pixel location and color index
    bounds = [(0, width-1), (0, height-1), (0, len(preset_colors)-1)]

    # Run the attack optimization
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

    # Extract the best perturbation found
    x, y, color_idx = int(result.x[0]), int(result.x[1]), int(result.x[2])
    r, g, b = preset_colors[color_idx % len(preset_colors)]

    return [x, y, r, g, b]

# def load_keras_model(h5_path):
#     """Loads a Keras model safely from an HDF5 file."""
#     with h5py.File(h5_path, 'r') as f:
#         model_config = f.attrs['model_config']
#         if isinstance(model_config, bytes):  # Old TensorFlow format
#             model_config = model_config.decode('utf-8')
#         model = model_from_json(model_config)  # Load model architecture
#         model.load_weights(h5_path)  # Load weights
#     return model

# # Load the model
# model = load_keras_model(MODEL_PATH)



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

def one_pixel_attack(image, preset_colors, max_iter=100):
    def perturbation(params):
        img_copy = image.copy()
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        response = call_model(img_copy)
        return -response["predictions"][0]["probability"]

    bounds = [(0, 64), (0, 64), (0, len(preset_colors)-0.001)]
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

    x, y, color_idx = int(result.x[0]), int(result.x[1]), int(result.x[2])
    r, g, b = preset_colors[color_idx % len(preset_colors)]
    return [x, y, r, g, b]

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

preset_colors = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # Based on research

api = call_modelapi(path)
print("API result: ", api)

optimal_pixel = one_pixel_attack(image, preset_colors)
# print("Optimal Pixel:", optimal_pixel)

optimal_api = one_pixel_attackapi(path, preset_colors)
new_api_image = produce_altered_image(image, optimal_pixel)
new_api = call_modelapi(new_api_image)
print("Pixel attack API", new_api)

altered = produce_altered_image(image, optimal_pixel)
new_prediction = call_model(altered)
print("Differential Evolution Prediction:", new_prediction)

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

