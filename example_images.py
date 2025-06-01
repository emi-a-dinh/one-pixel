# api_de_v2.py
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

def load_random_images(directory, num_samples=5000):
    """Selects a specified number of random images from the directory."""
    all_images = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def get_temp_path(prefix="temp"):
    """Generate a unique temporary file path to avoid collisions between processes."""
    os.makedirs("temp_files", exist_ok=True)
    return f"temp_files/{prefix}_{uuid.uuid4().hex}.png"

def differential_evolution_attack(image_path, popsize=20, generations=20, confidence_threshold=0.1):
    """Uses differential evolution to find optimal pixel modifications."""
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)
    h, w, _ = img_array.shape

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
        fitness_scores = []
        for candidate in population:
            x, y, r, g, b = candidate
            perturbed_image = img_array.copy()
            perturbed_image[y, x] = [r, g, b]
            temp_path = get_temp_path(f"de_{generation}")
            try:
                Image.fromarray(perturbed_image).save(temp_path)
                adv_prob = call_api_model(temp_path)
                fitness = abs(adv_prob - target)
                fitness_scores.append((fitness, candidate, adv_prob, perturbed_image))
            except Exception as e:
                print(f"Error evaluating candidate: {e}")
                fitness_scores.append((1.0, candidate, original_prob, img_array.copy()))
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            if fitness < confidence_threshold:
                best_solution = candidate
                best_fitness = fitness
                best_adv_prob = adv_prob
                best_perturbed_image = perturbed_image
                print(f"Early stopping at generation {generation}: Found solution with fitness {fitness}")
                break

        if best_solution is not None and best_fitness < confidence_threshold:
            break

        fitness_scores.sort(key=lambda x: x[0])
        current_best_fitness, current_best_solution, current_best_prob, current_best_image = fitness_scores[0]
        if current_best_fitness < best_fitness or best_solution is None:
            best_solution = current_best_solution
            best_fitness = current_best_fitness
            best_adv_prob = current_best_prob
            best_perturbed_image = current_best_image

        new_population = [fitness_scores[0][1]]
        while len(new_population) < popsize:
            a, b, c = random.sample(population, 3)
            mutant = []
            for i in range(5):
                if i < 2:  # x and y coordinates
                    value = int(a[i] + 0.8 * (b[i] - c[i]))
                    value = max(0, min(value, w-1 if i == 0 else h-1))
                else:  # RGB values
                    value = int(a[i] + 0.8 * (b[i] - c[i]))
                    value = max(0, min(value, 255))
                mutant.append(value)
            target_vector = random.choice(population)
            trial = []
            for i in range(5):
                if random.random() < 0.7:
                    trial.append(mutant[i])
                else:
                    trial.append(target_vector[i])
            new_population.append(trial)
        population = new_population
        print(f"Generation {generation + 1}/{generations}: Best fitness = {best_fitness}, Prob = {best_adv_prob}")

    adv_image = Image.fromarray(best_perturbed_image)
    adv_image_path = image_path.replace(".jpg", "_adv_de.jpg").replace(".png", "_adv_de.png")
    adv_image.save(adv_image_path)

    print(f"DE Attack - Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f}")
    flipped = 1 if ((original_prob >= 0.5 and best_adv_prob < 0.5) or
                    (original_prob < 0.5 and best_adv_prob >= 0.5)) else 0

    return {
        'method': 'differential_evolution',
        'original_prob': original_prob,
        'adversarial_prob': best_adv_prob,
        'flipped': flipped,
        'confidence_change': abs(best_adv_prob - original_prob)
    }

def multi_pixel_attack(image_path, num_pixels=1):
    """Applies a multi-pixel attack using a greedy search.
       num_pixels: number of pixels to modify (e.g., 1, 2, or 3)
    """
    from PIL import Image  # ensure PIL is imported
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image, dtype=np.uint8)
    h, w, _ = img_array.shape

    original_prob = call_api_model(image_path)
    target = 0.0 if original_prob > 0.5 else 1.0

    best_adv_prob = original_prob
    perturbed_image = img_array.copy()
    modified_pixels = []

    for pixel_idx in range(num_pixels):
        best_pixel = None
        best_color = None
        best_prob = best_adv_prob

        candidates = [(random.randint(0, w-1), random.randint(0, h-1)) for _ in range(100)]
        candidates = [p for p in candidates if p not in modified_pixels]

        for pixel in candidates:
            x, y = pixel
            for _ in range(10):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                temp_image = perturbed_image.copy()
                temp_image[y, x] = [r, g, b]
                temp_path = get_temp_path(f"multi_{pixel_idx}")
                try:
                    Image.fromarray(temp_image).save(temp_path)
                    adv_prob = call_api_model(temp_path)
                    if abs(adv_prob - target) < abs(best_prob - target):
                        best_prob = adv_prob
                        best_pixel = (x, y)
                        best_color = [r, g, b]
                except Exception as e:
                    print(f"Error testing pixel: {e}")
                finally:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except:
                        pass
                if ((original_prob >= 0.5 and best_prob < 0.5) or
                    (original_prob < 0.5 and best_prob >= 0.5)):
                    break

        if best_pixel is not None:
            x, y = best_pixel
            perturbed_image[y, x] = best_color
            modified_pixels.append(best_pixel)
            best_adv_prob = best_prob
            print(f"Pixel {pixel_idx+1}/{num_pixels}: Changed pixel at {best_pixel}, new prob: {best_adv_prob:.4f}")
        else:
            print(f"No improvement found for pixel {pixel_idx+1}")

    adv_image = Image.fromarray(perturbed_image)
    adv_image_path = image_path.replace(".jpg", f"_adv_multi_{num_pixels}.jpg").replace(".png", f"_adv_multi_{num_pixels}.png")
    adv_image.save(adv_image_path)

    print(f"Multi-Pixel Attack ({num_pixels} pixel(s)) - Original: {original_prob:.4f} → Adversarial: {best_adv_prob:.4f} (Modified {len(modified_pixels)} pixels)")
    flipped = 1 if ((original_prob >= 0.5 and best_adv_prob < 0.5) or
                    (original_prob < 0.5 and best_adv_prob >= 0.5)) else 0

    return {
        'method': f'multi_pixel_{num_pixels}',
        'original_prob': original_prob,
        'adversarial_prob': best_adv_prob,
        'flipped': flipped,
        'confidence_change': abs(best_adv_prob - original_prob),
        'pixels_modified': len(modified_pixels)
    }


def run_attack_with_comparison(image_path, methods=["de", "multi"]):
    """Run attacks on a single image and save the best multi-pixel result along with original and per-pixel versions."""
    results = []
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)

    best_multi_result = None
    best_multi_conf_change = -1
    multi_pixel_paths = {}

    try:
        if "de" in methods:
            print(f"\nRunning differential evolution attack on {filename}...")
            de_result = differential_evolution_attack(image_path)
            results.append(de_result)
    except Exception as e:
        print(f"DE attack failed on {filename}: {e}")

    if "multi" in methods:
        for npix in [1, 2, 3]:
            try:
                print(f"\nRunning multi-pixel attack ({npix} pixel{'s' if npix > 1 else ''}) on {filename}...")
                mp_result = multi_pixel_attack(image_path, num_pixels=npix)
                results.append(mp_result)

                attack_img_path = image_path.replace(ext, f"_adv_multi_{npix}{ext}")
                multi_pixel_paths[npix] = attack_img_path

                if mp_result["confidence_change"] > best_multi_conf_change:
                    best_multi_conf_change = mp_result["confidence_change"]
                    best_multi_result = {
                        "pixels": npix,
                        "file": attack_img_path
                    }
            except Exception as e:
                print(f"Multi-pixel attack ({npix} pixel) failed on {filename}: {e}")

    # Save original and each multi-pixel attack to new folder
    save_dir = os.path.join("best_pixel_attacks", base_name)
    os.makedirs(save_dir, exist_ok=True)

    try:
        # Save original image
        original_img = Image.open(image_path).convert("RGB")
        original_img.save(os.path.join(save_dir, f"original{ext}"))
    except Exception as e:
        print(f"Failed to save original image for {filename}: {e}")

    # Save all 1-, 2-, and 3-pixel adversarial images if they exist
    for npix in [1, 2, 3]:
        attack_img_path = multi_pixel_paths.get(npix)
        if attack_img_path and os.path.exists(attack_img_path):
            try:
                attacked_img = Image.open(attack_img_path)
                attacked_img.save(os.path.join(save_dir, f"multi_pixel_{npix}{ext}"))
            except Exception as e:
                print(f"Failed to save multi-pixel {npix} attack for {filename}: {e}")

    return results



def main():
    parser = argparse.ArgumentParser(description="Run and compare adversarial pixel attacks via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")
    parser.add_argument("--methods", type=str, default="both", choices=["de", "multi", "both"],
                        help="Attack methods to compare: differential evolution (de), multi-pixel (multi), or both")
    parser.add_argument("--samples", type=int, default=5000, help="Number of images to process")
    parser.add_argument("--pixels", type=int, default=5, help="(Legacy) Number of pixels to modify for multi-pixel attack")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes (use 1 for reliability)")

    args = parser.parse_args()
    image_dir = args.image_directory

    methods_to_run = []
    if args.methods == "de":
        methods_to_run = ["de"]
    elif args.methods == "multi":
        methods_to_run = ["multi"]
    else:
        methods_to_run = ["de", "multi"]

    os.makedirs("temp_files", exist_ok=True)
    image_paths = load_random_images(image_dir, args.samples)
    all_results = []

    print(f"Processing {len(image_paths)} images with methods: {methods_to_run}...")
    if args.parallel > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(
                    executor.submit(run_attack_with_comparison, image_path, methods_to_run)
                )
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    results = future.result()
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    print(f"Error in worker process: {e}")
    else:
        for image_path in tqdm(image_paths):
            results = run_attack_with_comparison(image_path, methods_to_run)
            if results:
                all_results.extend(results)

    for filename in os.listdir("temp_files"):
        try:
            os.remove(os.path.join("temp_files", filename))
        except:
            pass

    # Group results by method for summary statistics
    methods_summary = {}
    for result in all_results:
        m = result['method']
        if m not in methods_summary:
            methods_summary[m] = []
        methods_summary[m].append(result)

    print("\n===== ATTACK METHODS COMPARISON =====")
    for method, results in methods_summary.items():
        orig_avg = np.mean([r['original_prob'] for r in results])
        adv_avg = np.mean([r['adversarial_prob'] for r in results])
        conf_change_avg = np.mean([r['confidence_change'] for r in results])
        flip_count = sum(r['flipped'] for r in results)
        total = len(results)
        detection_rate = flip_count / total * 100 if total > 0 else 0
        print(f"\nMethod: {method}")
        print(f"  Images tested: {total}")
        print(f"  Average original confidence: {orig_avg:.4f}")
        print(f"  Average adversarial confidence: {adv_avg:.4f}")
        print(f"  Average confidence change: {conf_change_avg:.4f}")
        print(f"  Successfully flipped predictions: {flip_count}/{total} ({detection_rate:.1f}%)")
        if "multi_pixel" in method:
            pixels_mod = np.mean([r.get('pixels_modified', 0) for r in results])
            print(f"  Average pixels modified: {pixels_mod:.1f}")

    # Direct method comparison for images that underwent both attacks
    de_results = [r for r in all_results if r['method'] == 'differential_evolution']
    multi_methods = [r for r in all_results if 'multi_pixel' in r['method']]
    if de_results and multi_methods:
        de_orig_set = {r['original_prob'] for r in de_results}
        multi_orig_set = {r['original_prob'] for r in multi_methods}
        common_images = de_orig_set.intersection(multi_orig_set)
        if common_images:
            print("\nDirect Method Comparison (same images):")
            print(f"Number of images with both attacks: {len(common_images)}")
            common_de = [r for r in de_results if r['original_prob'] in common_images]
            common_multi = [r for r in multi_methods if r['original_prob'] in common_images]
            de_changes = [r['confidence_change'] for r in common_de]
            multi_changes = [r['confidence_change'] for r in common_multi]
            print(f"DE average confidence change: {np.mean(de_changes):.4f}")
            print(f"Multi-pixel average confidence change: {np.mean(multi_changes):.4f}")
            de_flips = sum(r['flipped'] for r in common_de)
            multi_flips = sum(r['flipped'] for r in common_multi)
            print(f"DE flipped predictions: {de_flips}/{len(common_de)} ({de_flips/len(common_de)*100:.1f}%)")
            print(f"Multi-pixel flipped predictions: {multi_flips}/{len(common_multi)} ({multi_flips/len(common_multi)*100:.1f}%)")

    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv("attack_comparison_results.csv", index=False)
        print("\nDetailed results saved to attack_comparison_results.csv")
    except ImportError:
        print("\nPandas not available, skipping CSV export")

if __name__ == "__main__":
    main()
