import os
import random
import numpy as np
import requests
import argparse

# API endpoint (replace with actual endpoint)
API_URL = "http://localhost:5000/model/predict"

def call_api_model(image_path):
    """Sends the image file to the API via multipart form-data."""
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        response = requests.post(API_URL, files=files)
        result = response.json()
        return result["predictions"][0]["probability"]

def load_random_images(directory, num_samples=1000):
    """Selects 100 random images from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Sends the image path to the API and gets the result."""
    prediction = call_api_model(image_path)
    # print(f"Prediction for {image_path}: {prediction}")
    return prediction

def main():
    parser = argparse.ArgumentParser(description="Run an image classification model via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")

    args = parser.parse_args()
    image_dir = args.image_directory

    image_paths = load_random_images(image_dir)

    results = [run_model_on_image(image_path) for image_path in image_paths]

    results = np.array(results)
    print("\n **Final Summary**")
    print(f"Mean of model outputs:\n {np.mean(results)}")
    print(f"Standard deviation of model outputs:\n {np.std(results)}")

if __name__ == "__main__":
    main()
