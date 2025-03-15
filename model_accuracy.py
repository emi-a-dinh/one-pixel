import os
import random
import numpy as np
import requests

# API endpoint (replace with actual endpoint)
API_URL = "http://0.0.0.0:5000/model/predict"

# Directory containing images
IMAGE_DIR = "../deep-histopath/data/mitoses/patches/val/mitosis"

def call_api_model(image_path):
    """Sends the image path to an API for inference."""
    data = {"image_path": image_path}  # API expects image path
    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        return response.json().get("prediction")  # Expecting a 'prediction' key in the response
    else:
        print(f"Error: API call failed for {image_path} with status {response.status_code}")
        return None

def load_random_images(directory, num_samples=100):
    """Selects 100 random images from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Sends the image path to the API and gets the result."""
    return call_api_model(image_path)  # API handles processing

def main():
    image_paths = load_random_images(IMAGE_DIR)
    results = []

    for image_path in image_paths:
        output = run_model_on_image(image_path)
        if output is not None:
            results.append(output)

    if results:
        results = np.array(results)  # Convert list to NumPy array
        mean_values = np.mean(results, axis=0)
        std_values = np.std(results, axis=0)

        print(f"Mean of model outputs:\n {mean_values}")
        print(f"Standard deviation of model outputs:\n {std_values}")

if __name__ == "__main__":
    main()
