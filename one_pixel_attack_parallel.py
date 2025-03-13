import requests
import numpy as np
from cv2 import imread, imwrite
from scipy.optimize import differential_evolution
from PIL import Image
import io
import os
import concurrent.futures
import argparse
import pandas as pd
import time
from tqdm import tqdm
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pixel_attack.log"),
        logging.StreamHandler()
    ]
)

MODEL = "http://0.0.0.0:5000/model/predict"
PRESET_COLORS = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # based on research

def call_model(image_array):
    """Send image to the model API and get predictions"""
    try:
        # Convert numpy array to bytes
        img_pil = Image.fromarray(np.uint8(image_array))
        
        # Ensure 64x64 size
        if img_pil.size != (64, 64):
            img_pil = img_pil.resize((64, 64))
        
        # Convert to PNG bytes
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Send to API
        files = {'image': ('image.png', buffer, 'image/png')}
        response = requests.post(MODEL, files=files, timeout=10)
        
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return {"predictions": [{"probability": 0}]}  # Default value on failure
    except Exception as e:
        logging.error(f"Error in call_model: {e}")
        return {"predictions": [{"probability": 0}]}

def one_pixel_attack(image, preset_colors, max_iter=100, patience=10):
    """Perform one-pixel attack with early stopping"""
    best_score = float('inf')
    patience_counter = 0
    
    def perturbation(params):
        nonlocal best_score, patience_counter
        
        img_copy = image.copy()
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        
        # Handle edge cases
        x = min(max(x, 0), img_copy.shape[1] - 1)
        y = min(max(y, 0), img_copy.shape[0] - 1)
        
        r, g, b = preset_colors[color_idx % len(preset_colors)]
        img_copy[y, x] = [b, g, r]  # OpenCV uses BGR order
        
        # Access the correct key path in the response
        response = call_model(img_copy)
        score = response["predictions"][0]["probability"]
        
        # Early stopping logic
        if score < best_score:
            best_score = score
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            return best_score
            
        return score

    bounds = [(0, 63), (0, 63), (0, len(preset_colors)-0.001)]
    result = differential_evolution(
        perturbation, 
        bounds, 
        maxiter=max_iter,
        popsize=10,
        disp=False,
        atol=1e-3
    )

    x, y, color_idx = int(result.x[0]), int(result.x[1]), int(result.x[2])
    r, g, b = preset_colors[color_idx % len(preset_colors)]
    
    # OpenCV uses BGR order, so we return the pixel info in BGR
    return [x, y, b, g, r]

def produce_altered_image(image, pixel):
    """Create a new image with the pixel attack applied"""
    altered_image = image.copy()
    x, y, b, g, r = map(int, pixel)
    
    # Ensure coordinates are within bounds
    x = min(max(x, 0), altered_image.shape[1] - 1)
    y = min(max(y, 0), altered_image.shape[0] - 1)
    
    altered_image[y, x] = [b, g, r]
    return altered_image

def process_single_image(image_path, original_dir, adversarial_dir, results_list, 
                         preset_colors=PRESET_COLORS, max_iter=100, image_type="normal"):
    """Process a single image for pixel attack"""
    try:
        # Extract filename
        filename = os.path.basename(image_path)
        
        # Read image
        image = imread(image_path)
        if image is None:
            logging.error(f"Failed to read image: {image_path}")
            return
            
        # Get original prediction
        original_pred = call_model(image)
        original_prob = original_pred.get("predictions", [{}])[0].get("probability", 0)
        
        # Run attack
        start_time = time.time()
        optimal_pixel = one_pixel_attack(image, preset_colors, max_iter=max_iter)
        attack_time = time.time() - start_time
        
        # Create altered image
        altered = produce_altered_image(image, optimal_pixel)
        
        # Get new prediction
        new_pred = call_model(altered)
        new_prob = new_pred.get("predictions", [{}])[0].get("probability", 0)
        
        # Copy original image to original directory
        original_output_path = os.path.join(original_dir, filename)
        imwrite(original_output_path, image)
        
        # Save altered image to adversarial directory
        adversarial_output_path = os.path.join(adversarial_dir, filename)
        imwrite(adversarial_output_path, altered)
        
        # Store results
        x, y, b, g, r = optimal_pixel
        results_list.append({
            "filename": filename,
            "type": image_type,
            "original_probability": original_prob,
            "new_probability": new_prob,
            "difference": original_prob - new_prob,
            "pixel_x": x,
            "pixel_y": y,
            "pixel_b": b,
            "pixel_g": g,
            "pixel_r": r,
            "attack_time": attack_time
        })
        
        logging.info(f"Processed {image_type} image {filename} - Original: {original_prob:.4f}, New: {new_prob:.4f}, Diff: {original_prob - new_prob:.4f}")
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def setup_directories(base_output_dir):
    """Set up directory structure for the experiment"""
    # Create main output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = {
        "normal_original": os.path.join(base_output_dir, "normal", "original"),
        "normal_adversarial": os.path.join(base_output_dir, "normal", "adversarial"),
        "cancer_original": os.path.join(base_output_dir, "cancer", "original"),
        "cancer_adversarial": os.path.join(base_output_dir, "cancer", "adversarial"),
        "results": os.path.join(base_output_dir, "results")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def batch_process_images(normal_input_dir, cancer_input_dir, output_dir, 
                         normal_limit=20000, cancer_limit=1000, 
                         max_workers=8, max_iterations=100):
    """Process normal and cancer images separately with specified limits"""
    
    # Setup directory structure
    dirs = setup_directories(output_dir)
    
    # Get normal image files (limit to 20,000)
    normal_files = [os.path.join(normal_input_dir, f) for f in os.listdir(normal_input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    normal_files = normal_files[:normal_limit]
    logging.info(f"Found {len(normal_files)} normal images to process (limit: {normal_limit})")
    
    # Get cancer image files (limit to 1,000)
    cancer_files = [os.path.join(cancer_input_dir, f) for f in os.listdir(cancer_input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    cancer_files = cancer_files[:cancer_limit]
    logging.info(f"Found {len(cancer_files)} cancer images to process (limit: {cancer_limit})")
    
    # Results container
    results = []
    
    # Process normal images in parallel
    logging.info("Processing normal images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_path in normal_files:
            future = executor.submit(
                process_single_image, 
                img_path, 
                dirs["normal_original"],
                dirs["normal_adversarial"],
                results, 
                PRESET_COLORS, 
                max_iterations,
                "normal"
            )
            futures.append(future)
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing normal images"):
            pass
    
    # Process cancer images in parallel
    logging.info("Processing cancer images...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_path in cancer_files:
            future = executor.submit(
                process_single_image, 
                img_path, 
                dirs["cancer_original"],
                dirs["cancer_adversarial"],
                results, 
                PRESET_COLORS, 
                max_iterations,
                "cancer"
            )
            futures.append(future)
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing cancer images"):
            pass
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(dirs["results"], "all_attack_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Split results by type and save separately
    normal_results = results_df[results_df["type"] == "normal"]
    cancer_results = results_df[results_df["type"] == "cancer"]
    
    normal_results.to_csv(os.path.join(dirs["results"], "normal_attack_results.csv"), index=False)
    cancer_results.to_csv(os.path.join(dirs["results"], "cancer_attack_results.csv"), index=False)
    
    # Generate summary statistics for each type
    normal_summary = {
        "type": "normal",
        "total_images": len(normal_results),
        "avg_original_prob": normal_results["original_probability"].mean(),
        "avg_new_prob": normal_results["new_probability"].mean(),
        "avg_difference": normal_results["difference"].mean(),
        "max_difference": normal_results["difference"].max(),
        "success_rate": (normal_results["difference"] > 0).mean() * 100,
        "avg_attack_time": normal_results["attack_time"].mean()
    }
    
    cancer_summary = {
        "type": "cancer",
        "total_images": len(cancer_results),
        "avg_original_prob": cancer_results["original_probability"].mean(),
        "avg_new_prob": cancer_results["new_probability"].mean(),
        "avg_difference": cancer_results["difference"].mean(),
        "max_difference": cancer_results["difference"].max(),
        "success_rate": (cancer_results["difference"] > 0).mean() * 100,
        "avg_attack_time": cancer_results["attack_time"].mean()
    }
    
    # Save summary
    summary_df = pd.DataFrame([normal_summary, cancer_summary])
    summary_path = os.path.join(dirs["results"], "attack_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logging.info(f"Batch processing complete. Results saved to {dirs['results']}")
    logging.info(f"Normal success rate: {normal_summary['success_rate']:.2f}%")
    logging.info(f"Cancer success rate: {cancer_summary['success_rate']:.2f}%")
    
    return results_df, summary_df, dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch One-Pixel Attack for Normal and Cancer Images')
    parser.add_argument('--normal_input', type=str, required=True, help='Input directory containing normal images')
    parser.add_argument('--cancer_input', type=str, required=True, help='Input directory containing cancer images')
    parser.add_argument('--output', type=str, required=True, help='Base output directory for all results')
    parser.add_argument('--normal_limit', type=int, default=20000, help='Number of normal images to process')
    parser.add_argument('--cancer_limit', type=int, default=1000, help='Number of cancer images to process')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--iterations', type=int, default=100, help='Max iterations for differential evolution')
    parser.add_argument('--model', type=str, default=None, help='Model endpoint URL (optional)')
    
    args = parser.parse_args()
    
    if args.model:
        MODEL = args.model
    
    start_time = time.time()
    results_df, summary_df, output_dirs = batch_process_images(
        args.normal_input, 
        args.cancer_input,
        args.output, 
        normal_limit=args.normal_limit,
        cancer_limit=args.cancer_limit,
        max_workers=args.workers,
        max_iterations=args.iterations
    )
    total_time = time.time() - start_time
    
    # Print summary for each type
    for _, row in summary_df.iterrows():
        img_type = row['type']
        print(f"\n{img_type.capitalize()} Images Summary:")
        print(f"Total images processed: {row['total_images']}")
        print(f"Average original probability: {row['avg_original_prob']:.4f}")
        print(f"Average new probability: {row['avg_new_prob']:.4f}")
        print(f"Average difference: {row['avg_difference']:.4f}")
        print(f"Maximum difference: {row['max_difference']:.4f}")
        print(f"Success rate: {row['success_rate']:.2f}%")
        print(f"Average attack time per image: {row['avg_attack_time']:.2f} seconds")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"\nFiles saved to:")
    for name, path in output_dirs.items():
        print(f"- {name}: {path}")