import os
import random
import numpy as np
import requests
import argparse
from PIL import Image

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
    """Selects a specified number of random images from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def run_model_on_image(image_path):
    """Sends the image path to the API and gets the result."""
    return call_api_model(image_path)

def fgsm_pixel_attack(image_path):
    """Performs a one-pixel FGSM attack to flip classification confidence above or below 0.5."""
    
    # Load original image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)

    # Get original prediction
    original_prob = run_model_on_image(image_path)

    # Choose a random pixel
    h, w, _ = img_array.shape
    x, y = random.randint(0, w - 1), random.randint(0, h - 1)

    # Determine attack direction
    perturbation = 10 if original_prob < 0.5 else -10  # Small pixel change
    img_array[y, x] = np.clip(img_array[y, x] + perturbation, 0, 255)  # Modify the pixel

    # Save adversarial image
    adv_image_path = image_path.replace(".jpg", "_adv.jpg").replace(".png", "_adv.png")
    adv_image = Image.fromarray(img_array)
    adv_image.save(adv_image_path)

    # Get new prediction
    adversarial_prob = run_model_on_image(adv_image_path)

    print(f"\nOriginal: {original_prob:.4f} → Adversarial: {adversarial_prob:.4f} (Pixel changed at ({x}, {y}))")

    return original_prob, adversarial_prob

def main():
    parser = argparse.ArgumentParser(description="Run an image classification model via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")

    args = parser.parse_args()
    image_dir = args.image_directory

    image_paths = load_random_images(image_dir)

    attack_results = [fgsm_pixel_attack(image_path) for image_path in image_paths]

    attack_results = np.array(attack_results)
    print("\nFinal Summary")
    print(f"Mean of original model outputs: {np.mean(attack_results[:, 0]):.4f}")
    print(f"Mean of adversarial model outputs: {np.mean(attack_results[:, 1]):.4f}")
    print(f"Standard deviation of model outputs: {np.std(attack_results, axis=0)}")

if __name__ == "__main__":
    main()
