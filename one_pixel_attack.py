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

MODELAPI = "http://0.0.0.0:5000/model/predict"

def call_modelapi(image_array):
    """Sends an image array to the API and returns the prediction."""
    if not isinstance(image_array, np.ndarray):
        raise ValueError("Invalid input: Provide a valid NumPy image array.")

    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray(np.uint8(image_array))

    # Resize to match model input size (if needed)
    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))

    # Convert image to bytes
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)

    # Send the image directly as a binary file
    files = {"image": ("image.png", buffer, "image/png")}
    response = requests.post(MODELAPI, files=files)

    return response.json()


# def call_modelapi(image_path):
#     # Ensure the input is a valid file path
#     if not isinstance(image_path, str) or not os.path.isfile(image_path):
#         raise ValueError("Invalid input: Provide a valid image file path.")

#     # Open the image file in binary mode and send it
#     with open(image_path, "rb") as img_file:
#         files = {"image": (image_path, img_file, "image/png")}
#         response = requests.post(MODELAPI, files=files)
    
#     return response.json()  # Return the JSON response


def one_pixel_attackapi(image_path, preset_colors, max_iter=100):
    """Performs one-pixel attack to either minimize or maximize probability."""
    image = cv2.imread(image_path)  # Load image using OpenCV (BGR format)

    if image is None:
        raise ValueError("Error loading image. Check the file path.")

    height, width, _ = image.shape  # Get image dimensions

    # Step 1: Get the original probability before perturbation
    original_response = call_modelapi(image_path)
    original_prob = original_response["predictions"][0]["probability"]

    # Decide attack direction:
    # - If probability is near 1, we want to DECREASE it (Minimize).
    # - If probability is near 0, we want to INCREASE it (Maximize).
    attack_direction = -1 if original_prob > 0.5 else 1  # Flip objective

    def perturbation(params):
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]

        # Make a copy and modify one pixel
        img_copy = image.copy()
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR format

        # Save perturbed image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            temp_path = temp_img.name
            cv2.imwrite(temp_path, img_copy)

        # Call the model API with the modified image
        response = call_modelapi(temp_path)

        # Remove the temporary image file after sending
        os.remove(temp_path)

        # Attack direction determines whether we minimize or maximize
        return attack_direction * response["predictions"][0]["probability"]

    # Define bounds for pixel position and color choice
    bounds = [(0, width - 1), (0, height - 1), (0, len(preset_colors) - 1)]

    # Run the optimization
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

    # Extract the best perturbation found
    x, y, color_idx = int(result.x[0]), int(result.x[1]), int(result.x[2])
    r, g, b = preset_colors[color_idx % len(preset_colors)]

    return [x, y, r, g, b]

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


def fgsm_attack_api(image, target_label=1, epsilon=0.1):
    """Performs FGSM attack using API calls instead of local model gradients."""
    
    image = image.astype(np.float32)  # Convert to float32
    perturbation = np.zeros_like(image)  # Initialize perturbation array
    delta = 1e-3  # Small perturbation for finite difference approximation

    # Compute numerical gradient for each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                perturbed_image = image.copy()
                perturbed_image[i, j, c] += delta  # Slightly increase pixel value

                original_pred = call_modelapi(image)["predictions"][0]["probability"]
                perturbed_pred = call_modelapi(perturbed_image)["predictions"][0]["probability"]

                # Compute approximate gradient (finite difference)
                gradient = (perturbed_pred - original_pred) / delta
                perturbation[i, j, c] = gradient

    # Apply adversarial perturbation in the direction of the gradient
    signed_grad = np.sign(perturbation)
    adversarial_image = image + epsilon * signed_grad  # Modify image
    adversarial_image = np.clip(adversarial_image, 0, 255).astype(np.uint8)  # Keep valid pixel values

    return adversarial_image


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

api = call_modelapi(image)
print("API result: ", api)


optimal_api = one_pixel_attackapi(path, preset_colors)
new_api_image = produce_altered_image(image, optimal_api)
imwrite("altered_image_api.png", new_api_image)
new_api = call_modelapi("altered_image_api.png")
print("Pixel attack API", new_api)

fggm_api = fgsm_attack_api(image)
new_fgsm_image = produce_altered_image(fggm_api, fggm_api)
imwrite("fgsm.png", new_fgsm_image)
new_fgsm = call_modelapi("fgsm.png")
print("Pixel attack api fgsm", new_fgsm)

# optimal_pixel = one_pixel_attack(image, preset_colors)
# print("Optimal Pixel:", optimal_pixel)


# altered = produce_altered_image(image, optimal_pixel)
# new_prediction = call_model(altered)
# print("Differential Evolution Prediction:", new_prediction)

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

