import os
import random
import numpy as np
import requests
import argparse
import cv2
from PIL import Image

# API endpoint
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

def get_salient_pixels(image_array, top_k=5):
    """Finds the most salient (important) pixels using edge detection."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Edge detection

    # Find non-zero edge pixels
    edge_pixels = np.argwhere(edges > 0)

    if len(edge_pixels) == 0:
        return [tuple(random.randint(0, i - 1) for i in image_array.shape[:2])]  # Fallback: random pixel

    # Randomly select top-k most salient pixels
    selected_pixels = random.sample(list(edge_pixels), min(top_k, len(edge_pixels)))
    return selected_pixels

def adaptive_pixel_attack(image_path, max_attempts=10, perturbation_magnitude=100):
    """Performs an adaptive attack by targeting the most salient pixels."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)

    # Get original prediction
    original_prob = call_api_model(image_path)

    # Find important pixels (high-contrast edges)
    salient_pixels = get_salient_pixels(img_array, top_k=5)

    best_adv_prob = original_prob
    best_pixel = None
    best_adv_image = img_array.copy()

    for _ in range(max_attempts):
        perturbed_image = img_array.copy()

        # Pick a high-impact pixel
        x, y = random.choice(salient_pixels)

        # Modify all RGB channels for maximum impact
        if original_prob < 0.5:
            perturbed_image[y, x] = np.clip(perturbed_image[y, x] + perturbation_magnitude, 0, 255)
        else:
            perturbed_image[y, x] = np.clip(perturbed_image[y, x] - perturbation_magnitude, 0, 255)

        # Save temporary adversarial image
        adv_image_path = image_path.replace(".jpg", "_adv.jpg").replace(".png", "_adv.png")
        adv_image = Image.fromarray(perturbed_image)
        adv_image.save(adv_image_path)

        # Get adversarial prediction
        adversarial_prob = call_api_model(adv_image_path)

        # Keep the best attack so far
        if abs(adversarial_prob - 0.5) > abs(best_adv_prob - 0.5):
            best_adv_prob = adversarial_prob
            best_pixel = (x, y)
            best_adv_image = perturbed_image.copy()

        # Stop early if the attack is successful
        if (original_prob >= 0.5 and best_adv_prob <= 0.5) or (original_prob < 0.5 and best_adv_prob > 0.5):
            break

    # Save the best adversarial image
    adv_image = Image.fromarray(best_adv_image)
    adv_image_path = image_path.replace(".jpg", "_adv_best.jpg").replace(".png", "_adv_best.png")
    adv_image.save(adv_image_path)

    print(f"Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Salient Pixel changed at {best_pixel})")

    return original_prob, best_adv_prob

def main():
    parser = argparse.ArgumentParser(description="Run an image classification model via API with adaptive pixel attack.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")

    args = parser.parse_args()
    image_dir = args.image_directory

    image_paths = load_random_images(image_dir)

    attack_results = [adaptive_pixel_attack(image_path) for image_path in image_paths]

    attack_results = np.array(attack_results)
    print("\nFinal Summary")
    print(f"Mean of original model outputs: {np.mean(attack_results[:, 0]):.4f}")
    print(f"Mean of adversarial model outputs: {np.mean(attack_results[:, 1]):.4f}")
    print(f"Standard deviation of model outputs: {np.std(attack_results, axis=0)}")

if __name__ == "__main__":
    main()
