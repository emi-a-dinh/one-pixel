import os
import random
import numpy as np
import requests
import argparse
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

def estimate_gradient(image_array, pixel, step=10):
    """Approximates the gradient by checking sensitivity to pixel changes."""
    x, y = pixel
    perturbed_image = image_array.copy()

    # Try increasing pixel intensity
    perturbed_image[y, x] = np.clip(perturbed_image[y, x] + step, 0, 255)
    Image.fromarray(perturbed_image).save("temp_plus.png")
    plus_prob = call_api_model("temp_plus.png")

    # Try decreasing pixel intensity
    perturbed_image[y, x] = np.clip(perturbed_image[y, x] - 2 * step, 0, 255)
    Image.fromarray(perturbed_image).save("temp_minus.png")
    minus_prob = call_api_model("temp_minus.png")

    # Restore original pixel
    perturbed_image[y, x] = image_array[y, x]

    # Compute numerical gradient (central difference approximation)
    gradient = (plus_prob - minus_prob) / (2 * step)
    return abs(gradient)

def fgsm_one_pixel_attack(image_path, max_attempts=20, perturbation_magnitude=255):
    """Applies FGSM-inspired one-pixel attack by estimating gradient information."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)

    # Get original prediction
    original_prob = call_api_model(image_path)

    h, w, _ = img_array.shape
    candidate_pixels = [(random.randint(0, w - 1), random.randint(0, h - 1)) for _ in range(50)]
    
    # Find the most sensitive pixel
    best_pixel = max(candidate_pixels, key=lambda p: estimate_gradient(img_array, p))
    
    best_adv_prob = original_prob
    best_adv_image = img_array.copy()

    for _ in range(max_attempts):
        perturbed_image = best_adv_image.copy()
        x, y = best_pixel

        # Modify pixel based on initial probability
        if original_prob < 0.5:
            perturbed_image[y, x] = np.clip(perturbed_image[y, x] + perturbation_magnitude, 0, 255)
        else:
            perturbed_image[y, x] = np.clip(perturbed_image[y, x] - perturbation_magnitude, 0, 255)

        # Save and test adversarial image
        adv_image_path = image_path.replace(".jpg", "_adv.jpg").replace(".png", "_adv.png")
        adv_image = Image.fromarray(perturbed_image)
        adv_image.save(adv_image_path)

        adversarial_prob = call_api_model(adv_image_path)

        # Keep the best attack so far
        if abs(adversarial_prob - 0.5) > abs(best_adv_prob - 0.5):
            best_adv_prob = adversarial_prob
            best_adv_image = perturbed_image.copy()

        # Stop early if the attack is successful
        if (original_prob >= 0.5 and best_adv_prob <= 0.5) or (original_prob < 0.5 and best_adv_prob > 0.5):
            break

    # Save final adversarial image
    adv_image = Image.fromarray(best_adv_image)
    adv_image_path = image_path.replace(".jpg", "_adv_best.jpg").replace(".png", "_adv_best.png")
    adv_image.save(adv_image_path)

    print(f"Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Pixel changed at {best_pixel})")

    return original_prob, best_adv_prob

def main():
    parser = argparse.ArgumentParser(description="Run an FGSM-inspired one-pixel attack via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")

    args = parser.parse_args()
    image_dir = args.image_directory

    image_paths = load_random_images(image_dir)

    attack_results = [fgsm_one_pixel_attack(image_path) for image_path in image_paths]

    attack_results = np.array(attack_results)
    print("\nFinal Summary")
    print(f"Mean of original model outputs: {np.mean(attack_results[:, 0]):.4f}")
    print(f"Mean of adversarial model outputs: {np.mean(attack_results[:, 1]):.4f}")
    print(f"Standard deviation of model outputs: {np.std(attack_results, axis=0)}")

if __name__ == "__main__":
    main()
