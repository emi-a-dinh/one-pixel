import numpy as np
import h5py
import tensorflow as tf
from cv2 import imread, imwrite
from scipy.optimize import differential_evolution
from PIL import Image
from tensorflow.keras.models import model_from_json
import io
import requests

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


def call_modelapi(image_array):
  
    img_pil = Image.fromarray(np.uint8(image_array))
 
    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    files = {'image': ('image.png', buffer, 'image/png')}
    response = requests.post(MODELAPI, files=files)
    
    return response.json()


def one_pixel_attackapi(image, preset_colors, max_iter=100):
    def perturbation(params):
        img_copy = image.copy()
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]
        img_copy[y, x] = [b, g, r]  
        
        response = call_model(img_copy)
        return -response["predictions"][0]["probability"]  

    bounds = [(0, 64), (0, 64), (0, len(preset_colors)-0.001)]
    result = differential_evolution(perturbation, bounds, maxiter=max_iter)

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


path = "1_02_21_1791_131_0_-14_-14_4.png"
image = imread(path)
original = call_model(image)
print("Original Prediction:", original)

preset_colors = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # Based on research

api = call_modelapi(image)
print("API result: ", api)

# optimal_pixel = one_pixel_attack(image, preset_colors)
# print("Optimal Pixel:", optimal_pixel)

# altered = produce_altered_image(image, optimal_pixel)
# new_prediction = call_model(altered)
# print("Differential Evolution Prediction:", new_prediction)

fgsm_image = fgsm_attack(image)
fgsm_prediction = call_model(fgsm_image)
print("New Prediction (after FGSM attack):", fgsm_prediction)
# imwrite("altered_image1.png", altered)

