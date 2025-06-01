#api_control.py
import os
import random
import numpy as np
import requests
import argparse
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import time
import pandas as pd

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
    all_images = [os.path.join(directory, f) for f in os.listdir(directory)
                  if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    return random.sample(all_images, min(num_samples, len(all_images)))

def classify_image(image_path):
    """Classify a single image using the API."""
    try:
        probability = call_api_model(image_path)
        prediction = "Positive" if probability >= 0.5 else "Negative"
        return {
            'image_path': image_path,
            'filename': os.path.basename(image_path),
            'probability': probability,
            'prediction': prediction
        }
    except Exception as e:
        print(f"Error classifying {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Baseline image classification via API.")
    parser.add_argument("image_directory", type=str, help="Path to the directory containing images")
    parser.add_argument("--samples", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes (use 1 for reliability)")
    parser.add_argument("--output", type=str, default="baseline_results.csv", help="Output CSV filename")

    args = parser.parse_args()
    
    print(f"Loading up to {args.samples} random images from {args.image_directory}...")
    image_paths = load_random_images(args.image_directory, args.samples)
    
    results = []
    start_time = time.time()
    
    if args.parallel > 1:
        print(f"Processing {len(image_paths)} images with {args.parallel} parallel workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
            futures = []
            for image_path in image_paths:
                futures.append(executor.submit(classify_image, image_path))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    results.append(result)
    else:
        print(f"Processing {len(image_paths)} images sequentially...")
        for image_path in tqdm(image_paths):
            result = classify_image(image_path)
            if result:
                results.append(result)
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    total_images = len(results)
    positive_count = sum(1 for r in results if r['prediction'] == 'Positive')
    negative_count = total_images - positive_count
    average_probability = np.mean([r['probability'] for r in results])
    
    # Calculate percentages
    positive_percentage = (positive_count / total_images) * 100 if total_images > 0 else 0
    negative_percentage = (negative_count / total_images) * 100 if total_images > 0 else 0
    
    print("\n===== BASELINE CLASSIFICATION RESULTS =====")
    print(f"Total images processed: {total_images}")
    print(f"Positive predictions: {positive_count} ({positive_percentage:.2f}%)")
    print(f"Negative predictions: {negative_count} ({negative_percentage:.2f}%)")
    print(f"Average confidence: {average_probability:.4f}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    
    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")
    else:
        print("\nNo results to save.")
    
    # Calculate additional statistics
    if results:
        prob_distribution = np.array([r['probability'] for r in results])
        print("\nProbability Distribution:")
        print(f"Min: {np.min(prob_distribution):.4f}")
        print(f"Max: {np.max(prob_distribution):.4f}")
        print(f"Median: {np.median(prob_distribution):.4f}")
        print(f"Std Dev: {np.std(prob_distribution):.4f}")
        
        # Calculate confidence ranges
        ranges = {
            "0.0-0.1": 0,
            "0.1-0.2": 0,
            "0.2-0.3": 0,
            "0.3-0.4": 0,
            "0.4-0.5": 0,
            "0.5-0.6": 0,
            "0.6-0.7": 0,
            "0.7-0.8": 0,
            "0.8-0.9": 0,
            "0.9-1.0": 0
        }
        
        for r in results:
            prob = r['probability']
            for range_str in ranges.keys():
                lower, upper = map(float, range_str.split('-'))
                if lower <= prob < upper or (upper == 1.0 and prob == 1.0):
                    ranges[range_str] += 1
                    break
        
        print("\nConfidence Distribution:")
        for range_str, count in ranges.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"{range_str}: {count} images ({percentage:.2f}%)")

if __name__ == "__main__":
    main()