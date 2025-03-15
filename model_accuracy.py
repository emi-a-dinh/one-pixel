import os
import random
import numpy as np
import requests
import argparse

# API endpoint (replace with actual endpoint)
API_URL = "http://localhost:5000/model/predict"  # Example endpoint

def call_api_model(image_path):
    """Sends the image file to the API via multipart form-data."""
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}  # Mimic 'curl -F "image=@image.png"'
        response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        return response.json().get("prediction")  # Expecting a 'prediction' key in the response
    else:
        print(f"Error: API call failed for {image_path} with status {response.status_code}")
        return None

def load_random_images(directory, num_samples=100):
    """Selects 100 random images from the directory."""
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return []

    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    
    if not all_images:
        print(f"Error: No images found in directory '{directory}'.")
        return []

    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Sends the image path to the API and gets the result."""
    return call_api_model(image_path)  # API handles processing

def main():
    parser = argparse.ArgumentParser(description="Run an image classification model via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")
    
    args = parser.parse_args()
    image_dir = args.image_directory

    image_paths = load_random_images(image_dir)
    
    if not image_paths:
        return  # Exit if no images were found

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
