import os
import random
import numpy as np
import requests
import argparse
from PIL import Image
from tqdm import tqdm
import concurrent.futures

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

def differential_evolution_attack(image_path, popsize=20, generations=20, confidence_threshold=0.1):
    """Uses differential evolution to find optimal pixel modifications."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)
    h, w, _ = img_array.shape
    
    # Get original prediction
    original_prob = call_api_model(image_path)
    target = 0.0 if original_prob > 0.5 else 1.0
    
    # Initialize population: [x, y, r, g, b]
    population = []
    for _ in range(popsize):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        population.append([x, y, r, g, b])
    
    best_solution = None
    best_fitness = float('inf')
    
    for generation in range(generations):
        # Evaluate fitness of each candidate
        fitness_scores = []
        for candidate in population:
            x, y, r, g, b = candidate
            perturbed_image = img_array.copy()
            perturbed_image[y, x] = [r, g, b]
            
            # Save and test adversarial image
            temp_path = f"temp_{generation}_{x}_{y}.png"
            Image.fromarray(perturbed_image).save(temp_path)
            adv_prob = call_api_model(temp_path)
            
            # Fitness is distance from target probability
            fitness = abs(adv_prob - target)
            fitness_scores.append((fitness, candidate, adv_prob, perturbed_image))
            
            # Clean up temporary file
            os.remove(temp_path)
            
            # Stop early if we've found a good solution
            if fitness < confidence_threshold:
                best_solution = candidate
                best_fitness = fitness
                best_adv_prob = adv_prob
                best_perturbed_image = perturbed_image
                print(f"Early stopping at generation {generation}: Found solution with fitness {fitness}")
                break
        
        if best_solution is not None and best_fitness < confidence_threshold:
            break
        
        # Sort by fitness and keep the best solution
        fitness_scores.sort(key=lambda x: x[0])
        current_best_fitness, current_best_solution, current_best_prob, current_best_image = fitness_scores[0]
        
        if current_best_fitness < best_fitness or best_solution is None:
            best_solution = current_best_solution
            best_fitness = current_best_fitness
            best_adv_prob = current_best_prob
            best_perturbed_image = current_best_image
            
        # Create new generation
        new_population = [fitness_scores[0][1]]  # Keep the best solution
        
        while len(new_population) < popsize:
            # Select three random distinct candidates
            a, b, c = random.sample(population, 3)
            
            # Create mutant
            mutant = []
            for i in range(5):
                if i < 2:  # x and y coordinates
                    value = int(a[i] + 0.8 * (b[i] - c[i]))
                    value = max(0, min(value, w-1 if i == 0 else h-1))
                else:  # RGB values
                    value = int(a[i] + 0.8 * (b[i] - c[i]))
                    value = max(0, min(value, 255))
                mutant.append(value)
            
            # Crossover with a random candidate
            target_vector = random.choice(population)
            trial = []
            for i in range(5):
                if random.random() < 0.7:  # Crossover probability
                    trial.append(mutant[i])
                else:
                    trial.append(target_vector[i])
            
            new_population.append(trial)
        
        population = new_population
        
        print(f"Generation {generation + 1}/{generations}: Best fitness = {best_fitness}, Prob = {best_adv_prob}")
    
    # Save final adversarial image
    adv_image = Image.fromarray(best_perturbed_image)
    adv_image_path = image_path.replace(".jpg", "_adv_best.jpg").replace(".png", "_adv_best.png")
    adv_image.save(adv_image_path)
    
    print(f"Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Pixel changed at {best_solution[0]}, {best_solution[1]})")
    
    return original_prob, best_adv_prob

def multi_pixel_attack(image_path, num_pixels=5, max_iterations=50):
    """Applies a multi-pixel attack using greedy search."""
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)
    h, w, _ = img_array.shape
    
    # Get original prediction
    original_prob = call_api_model(image_path)
    target = 0.0 if original_prob > 0.5 else 1.0
    
    best_adv_prob = original_prob
    perturbed_image = img_array.copy()
    modified_pixels = []
    
    for pixel_idx in range(num_pixels):
        best_pixel = None
        best_color = None
        best_prob = best_adv_prob
        
        # Try a set of random pixels
        candidates = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(min(100, max_iterations))]
        
        # Remove pixels that have already been modified
        candidates = [p for p in candidates if p not in modified_pixels]
        
        for pixel in candidates:
            x, y = pixel
            
            # Try a set of random colors
            for _ in range(10):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                
                # Apply perturbation
                temp_image = perturbed_image.copy()
                temp_image[y, x] = [r, g, b]
                
                # Save and test
                temp_path = f"temp_multi_{pixel_idx}_{x}_{y}.png"
                Image.fromarray(temp_image).save(temp_path)
                adv_prob = call_api_model(temp_path)
                os.remove(temp_path)
                
                # Update best if this is better
                if abs(adv_prob - target) < abs(best_prob - target):
                    best_prob = adv_prob
                    best_pixel = (x, y)
                    best_color = [r, g, b]
                    
                # Stop early if we've flipped the prediction
                if (original_prob >= 0.5 and adv_prob < 0.5) or (original_prob < 0.5 and adv_prob >= 0.5):
                    break
        
        # If we found a better pixel, update our image
        if best_pixel is not None:
            x, y = best_pixel
            perturbed_image[y, x] = best_color
            modified_pixels.append(best_pixel)
            best_adv_prob = best_prob
            print(f"Pixel {pixel_idx+1}/{num_pixels}: Changed pixel at {best_pixel}, new prob: {best_adv_prob:.4f}")
        else:
            print(f"No improvement found for pixel {pixel_idx+1}")
    
    # Save final adversarial image
    adv_image = Image.fromarray(perturbed_image)
    adv_image_path = image_path.replace(".jpg", "_adv_multi.jpg").replace(".png", "_adv_multi.png")
    adv_image.save(adv_image_path)
    
    print(f"Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Modified {len(modified_pixels)} pixels)")
    
    return original_prob, best_adv_prob

def parallel_attack(image_path):
    """Try both attack methods and return the best result."""
    de_result = differential_evolution_attack(image_path)
    mp_result = multi_pixel_attack(image_path)
    
    # Return the result that changed the probability the most
    orig_prob = de_result[0]  # Both should have the same original probability
    de_adv_prob = de_result[1]
    mp_adv_prob = mp_result[1]
    
    if abs(de_adv_prob - 0.5) < abs(mp_adv_prob - 0.5):
        return orig_prob, de_adv_prob
    else:
        return orig_prob, mp_adv_prob

def main():
    parser = argparse.ArgumentParser(description="Run improved adversarial pixel attacks via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")
    parser.add_argument("--method", type=str, default="best", choices=["de", "multi", "best"],
                        help="Attack method: differential evolution (de), multi-pixel (multi), or best of both (best)")
    parser.add_argument("--samples", type=int, default=20, help="Number of images to process")
    parser.add_argument("--pixels", type=int, default=5, help="Number of pixels to modify for multi-pixel attack")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes")
    
    args = parser.parse_args()
    image_dir = args.image_directory
    
    image_paths = load_random_images(image_dir, args.samples)
    attack_results = []
    
    print(f"Processing {len(image_paths)} images...")
    
    if args.parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            attack_func = {
                "de": differential_evolution_attack,
                "multi": lambda path: multi_pixel_attack(path, num_pixels=args.pixels),
                "best": parallel_attack
            }[args.method]
            
            attack_results = list(tqdm(executor.map(attack_func, image_paths), total=len(image_paths)))
    else:
        for image_path in tqdm(image_paths):
            if args.method == "de":
                result = differential_evolution_attack(image_path)
            elif args.method == "multi":
                result = multi_pixel_attack(image_path, num_pixels=args.pixels)
            else:  # best
                result = parallel_attack(image_path)
            attack_results.append(result)
    
    attack_results = np.array(attack_results)
    
    print("\nFinal Summary")
    print(f"Mean of original model outputs: {np.mean(attack_results[:, 0]):.4f}")
    print(f"Mean of adversarial model outputs: {np.mean(attack_results[:, 1]):.4f}")
    print(f"Standard deviation of model outputs: {np.std(attack_results, axis=0)}")
    
    # Count how many predictions were flipped
    flipped = sum(1 for orig, adv in attack_results if (orig >= 0.5 and adv < 0.5) or (orig < 0.5 and adv >= 0.5))
    print(f"Successfully flipped {flipped}/{len(attack_results)} predictions ({flipped/len(attack_results)*100:.1f}%)")

if __name__ == "__main__":
    main()