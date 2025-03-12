import requests
import numpy as np
from cv2 import imread, imwrite
from scipy.optimize import differential_evolution
from PIL import Image
import io


MODEL = "http://0.0.0.0:5000/model/predict"

""" TODO: need to ensure that this is how to format the payload"""
def call_model(image_array):
    # Convert numpy array to bytes
    img_pil = Image.fromarray(np.uint8(image_array))
    
    # Ensure 64x64 size
    if img_pil.size != (64, 64):
        img_pil = img_pil.resize((64, 64))
    
    # Convert to PNG bytes
    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Send to API
    files = {'image': ('image.png', buffer, 'image/png')}
    response = requests.post(MODEL, files=files)
    
    return response.json()


def one_pixel_attack(image, preset_colors, max_iter=100):
    def perturbation(params):
        img_copy = image.copy()
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        r, g, b = preset_colors[color_idx % len(preset_colors)]
        img_copy[y, x] = [r, g, b] 
        return call_model(img_copy)["confidence"] 

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
print("original :", original)

preset_colors = [[0,0,0], [255,255,255], [255, 255, 0]] # based on research

optimal_pixel = one_pixel_attack(image, preset_colors)
print("optimal pixel:", optimal_pixel)

altered = produce_altered_image(image, optimal_pixel)
new_prediction = call_model(altered)
print("New Prediction:", new_prediction)

imwrite("altered_image.png", altered)
