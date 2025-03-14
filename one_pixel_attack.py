import numpy as np
import h5py
import tensorflow as tf
from cv2 import imread, imwrite
from scipy.optimize import differential_evolution
from PIL import Image
from tensorflow.keras.models import model_from_json


MODEL_PATH = "0.29452_f1max_0.14705_f1_0.78622_loss_0_epoch_model.hdf5"  # Path to your HDF5 model file

def load_keras_model(h5_path):
    """Loads a Keras model safely from an HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs['model_config']
        if isinstance(model_config, bytes):  # Old TensorFlow format
            model_config = model_config.decode('utf-8')
        model = model_from_json(model_config)  # Load model architecture
        model.load_weights(h5_path)  # Load weights
    return model

model = load_keras_model(MODEL_PATH)

def call_model(image_array):
    """Runs the local HDF5 model on the input image."""
    img_pil = Image.fromarray(np.uint8(image_array))

    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))

    img_np = np.array(img_pil) / 255.0  # Normalize pixel values
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

    predictions = model.predict(img_np)
    return {"predictions": [{"probability": float(predictions[0][0])}]}  # Assuming binary classification


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


def produce_altered_image(image, pixel):
    altered_image = image.copy()
    x, y, r, g, b = map(int, pixel)
    altered_image[y, x] = [r, g, b]

    return altered_image


path = "1_01_01_0_0_0_0_0_.png"
image = imread(path)
original = call_model(image)
print("Original Prediction:", original)

preset_colors = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # Based on research

optimal_pixel = one_pixel_attack(image, preset_colors)
print("Optimal Pixel:", optimal_pixel)

altered = produce_altered_image(image, optimal_pixel)
new_prediction = call_model(altered)
print("New Prediction:", new_prediction)

imwrite("altered_image1.png", altered)
