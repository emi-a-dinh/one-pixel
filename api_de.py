import os
import random
import numpy as np
import requests
import argparse
import uuid
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import time

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

def get_temp_path(prefix="temp"):
    """Generate a unique temporary file path to avoid collisions between processes."""
    # Create a temp directory if it doesn't exist
    os.makedirs("temp_files", exist_ok=True)
    return f"temp_files/{prefix}_{uuid.uuid4().hex}.png"

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
    best_adv_prob = original_prob
    best_perturbed_image = img_array.copy()
    
    for generation in range(generations):
        # Evaluate fitness of each candidate
        fitness_scores = []
        for candidate in population:
            x, y, r, g, b = candidate
            perturbed_image = img_array.copy()
            perturbed_image[y, x] = [r, g, b]
            
            # Save and test adversarial image
            temp_path = get_temp_path(f"de_{generation}")
            try:
                Image.fromarray(perturbed_image).save(temp_path)
                adv_prob = call_api_model(temp_path)
                
                # Fitness is distance from target probability
                fitness = abs(adv_prob - target)
                fitness_scores.append((fitness, candidate, adv_prob, perturbed_image))
                
            except Exception as e:
                print(f"Error evaluating candidate: {e}")
                # Use a high fitness value for failed evaluations
                fitness_scores.append((1.0, candidate, original_prob, img_array.copy()))
            
            finally:
                # Clean up temporary file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            
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
    adv_image_path = image_path.replace(".jpg", "_adv_de.jpg").replace(".png", "_adv_de.png")
    adv_image.save(adv_image_path)
    
    print(f"DE Attack - Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f}")
    
    flipped = 1 if (original_prob >= 0.5 and best_adv_prob < 0.5) or (original_prob < 0.5 and best_adv_prob >= 0.5) else 0
    
    return {
        'method': 'differential_evolution',
        'original_prob': original_prob,
        'adversarial_prob': best_adv_prob,
        'flipped': flipped,
        'confidence_change': abs(best_adv_prob - original_prob)
    }

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
                temp_path = get_temp_path(f"multi_{pixel_idx}")
                try:
                    Image.fromarray(temp_image).save(temp_path)
                    adv_prob = call_api_model(temp_path)
                    
                    # Update best if this is better
                    if abs(adv_prob - target) < abs(best_prob - target):
                        best_prob = adv_prob
                        best_pixel = (x, y)
                        best_color = [r, g, b]
                except Exception as e:
                    print(f"Error testing pixel: {e}")
                finally:
                    # Clean up
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
                
                # Stop early if we've flipped the prediction
                if (original_prob >= 0.5 and best_prob < 0.5) or (original_prob < 0.5 and best_prob >= 0.5):
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
    
    print(f"Multi-pixel Attack - Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Modified {len(modified_pixels)} pixels)")
    
    flipped = 1 if (original_prob >= 0.5 and best_adv_prob < 0.5) or (original_prob < 0.5 and best_adv_prob >= 0.5) else 0
    
    return {
        'method': 'multi_pixel',
        'original_prob': original_prob,
        'adversarial_prob': best_adv_prob,
        'flipped': flipped,
        'confidence_change': abs(best_adv_prob - original_prob),
        'pixels_modified': len(modified_pixels)
    }

def run_attack_with_comparison(image_path, methods=["de", "multi"], num_pixels=5):
    """Run multiple attack methods on a single image and return results for comparison."""
    results = []
    filename = os.path.basename(image_path)
    
    try:
        if "de" in methods:
            print(f"\nRunning differential evolution attack on {filename}...")
            de_result = differential_evolution_attack(image_path)
            results.append(de_result)
    except Exception as e:
        print(f"DE attack failed on {filename}: {e}")
    
    try:
        if "multi" in methods:
            print(f"\nRunning multi-pixel attack on {filename}...")
            mp_result = multi_pixel_attack(image_path, num_pixels=num_pixels)
            results.append(mp_result)
    except Exception as e:
        print(f"Multi-pixel attack failed on {filename}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run and compare adversarial pixel attacks via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")
    parser.add_argument("--methods", type=str, default="both", choices=["de", "multi", "both"],
                       help="Attack methods to compare: differential evolution (de), multi-pixel (multi), or both")
    parser.add_argument("--samples", type=int, default=10, help="Number of images to process")
    parser.add_argument("--pixels", type=int, default=5, help="Number of pixels to modify for multi-pixel attack")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes (use 1 for reliability)")
    
    args = parser.parse_args()
    image_dir = args.image_directory
    
    # Determine which methods to run
    methods_to_run = []
    if args.methods == "de":
        methods_to_run = ["de"]
    elif args.methods == "multi":
        methods_to_run = ["multi"]
    else:
        methods_to_run = ["de", "multi"]
    
    # Create temp directory
    os.makedirs("temp_files", exist_ok=True)
    
    image_paths = load_random_images(image_dir, args.samples)
    all_results = []
    
    print(f"Processing {len(image_paths)} images with methods: {methods_to_run}...")
    
    if args.parallel > 1:
        # More robust parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(
                    executor.submit(run_attack_with_comparison, image_path, methods_to_run, args.pixels)
                )
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    print(f"Error in worker process: {e}")
    else:
        # Sequential processing
        for image_path in tqdm(image_paths):
            results = run_attack_with_comparison(image_path, methods_to_run, args.pixels)
            if results:
                all_results.extend(results)
    
    # Clean up temp directory
    for filename in os.listdir("temp_files"):
        try:
            os.remove(os.path.join("temp_files", filename))
        except:
            pass
    
    # Group results by method
    de_results = [r for r in all_results if r['method'] == 'differential_evolution']
    mp_results = [r for r in all_results if r['method'] == 'multi_pixel']
    
    # Statistics for comparison
    print("\n===== ATTACK METHODS COMPARISON =====")
    
    print("\nDifferential Evolution Attack:")
    print(f"Images tested: {len(de_results)}")
    print(f"Average original confidence: {np.mean([r['original_prob'] for r in de_results]):.4f}")
    print(f"Average adversarial confidence: {np.mean([r['adversarial_prob'] for r in de_results]):.4f}")
    print(f"Average confidence change: {np.mean([r['confidence_change'] for r in de_results]):.4f}")
    
    flipped = sum(r['flipped'] for r in de_results)
    print(f"Successfully flipped predictions: {flipped}/{len(de_results)} ({flipped/len(de_results)*100:.1f}%)")
    
    if mp_results:
        print("\nMulti-Pixel Attack:")
        print(f"Images tested: {len(mp_results)}")
        print(f"Average original confidence: {np.mean([r['original_prob'] for r in mp_results]):.4f}")
        print(f"Average adversarial confidence: {np.mean([r['adversarial_prob'] for r in mp_results]):.4f}")
        print(f"Average confidence change: {np.mean([r['confidence_change'] for r in mp_results]):.4f}")
        
        flipped = sum(r['flipped'] for r in mp_results)
        print(f"Successfully flipped predictions: {flipped}/{len(mp_results)} ({flipped/len(mp_results)*100:.1f}%)")
        
        if 'pixels_modified' in mp_results[0]:
            print(f"Average pixels modified: {np.mean([r.get('pixels_modified', 0) for r in mp_results]):.1f}")
    
    # Direct comparison for images that had both attacks
    if de_results and mp_results:
        # Find images that were tested with both methods
        de_images = {r['original_prob'] for r in de_results}  # Using original prob as a proxy for image ID
        mp_images = {r['original_prob'] for r in mp_results}
        common_images = de_images.intersection(mp_images)
        
        if common_images:
            print("\nDirect Method Comparison (same images):")
            print(f"Number of images with both attacks: {len(common_images)}")
            
            # Filter results for common images
            common_de = [r for r in de_results if r['original_prob'] in common_images]
            common_mp = [r for r in mp_results if r['original_prob'] in common_images]
            
            # Compare effectiveness
            de_changes = [r['confidence_change'] for r in common_de]
            mp_changes = [r['confidence_change'] for r in common_mp]
            
            print(f"DE average confidence change: {np.mean(de_changes):.4f}")
            print(f"Multi-pixel average confidence change: {np.mean(mp_changes):.4f}")
            
            de_flips = sum(r['flipped'] for r in common_de)
            mp_flips = sum(r['flipped'] for r in common_mp)
            
            print(f"DE flipped predictions: {de_flips}/{len(common_de)} ({de_flips/len(common_de)*100:.1f}%)")
            print(f"Multi-pixel flipped predictions: {mp_flips}/{len(common_mp)} ({mp_flips/len(common_mp)*100:.1f}%)")
            
            # Count cases where one method outperformed the other
            de_wins = 0
            mp_wins = 0
            ties = 0
            
            for de_r in common_de:
                # Find matching multi-pixel result
                mp_r = next((r for r in common_mp if r['original_prob'] == de_r['original_prob']), None)
                if mp_r:
                    if de_r['confidence_change'] > mp_r['confidence_change']:
                        de_wins += 1
                    elif mp_r['confidence_change'] > de_r['confidence_change']:
                        mp_wins += 1
                    else:
                        ties += 1
            
            print(f"DE performed better: {de_wins} images ({de_wins/len(common_images)*100:.1f}%)")
            print(f"Multi-pixel performed better: {mp_wins} images ({mp_wins/len(common_images)*100:.1f}%)")
            print(f"Both methods equally effective: {ties} images ({ties/len(common_images)*100:.1f}%)")
    
    # Save results to CSV for further analysis
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv("attack_comparison_results.csv", index=False)
        print("\nDetailed results saved to attack_comparison_results.csv")
    except ImportError:
        print("\nPandas not available, skipping CSV export")

if __name__ == "__main__":
    main()
