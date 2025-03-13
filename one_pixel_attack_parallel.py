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

def process_single_image(image_path, output_dir, results_list, preset_colors=PRESET_COLORS, max_iter=100):
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
        
        # Save altered image
        output_path = os.path.join(output_dir, f"altered_{filename}")
        imwrite(output_path, altered)
        
        # Store results
        x, y, b, g, r = optimal_pixel
        results_list.append({
            "filename": filename,
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
        
        logging.info(f"Processed {filename} - Original: {original_prob:.4f}, New: {new_prob:.4f}, Diff: {original_prob - new_prob:.4f}")
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")

def batch_process_images(input_dir, output_dir, max_workers=8, max_iterations=100, batch_size=None):
    """Process multiple images in parallel batches"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if batch_size:
        image_files = image_files[:batch_size]
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Results container
    results = []
    
    # Process images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for img_path in image_files:
            future = executor.submit(
                process_single_image, 
                img_path, 
                output_dir, 
                results, 
                PRESET_COLORS, 
                max_iterations
            )
            futures.append(future)
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing images"):
            pass
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "attack_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Generate summary statistics
    summary = {
        "total_images": len(results),
        "avg_original_prob": results_df["original_probability"].mean(),
        "avg_new_prob": results_df["new_probability"].mean(),
        "avg_difference": results_df["difference"].mean(),
        "max_difference": results_df["difference"].max(),
        "success_rate": (results_df["difference"] > 0).mean() * 100,
        "avg_attack_time": results_df["attack_time"].mean()
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(output_dir, "attack_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    logging.info(f"Batch processing complete. Results saved to {results_path}")
    logging.info(f"Success rate: {summary['success_rate']:.2f}%")
    
    return results_df, summary_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch One-Pixel Attack')
    parser.add_argument('--input', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for altered images and results')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--iterations', type=int, default=100, help='Max iterations for differential evolution')
    parser.add_argument('--batch', type=int, default=None, help='Limit processing to this number of images')
    parser.add_argument('--model', type=str, default=None, help='Model endpoint URL (optional)')
    
    args = parser.parse_args()
    
    if args.model:
        MODEL = args.model
    
    start_time = time.time()
    results_df, summary_df = batch_process_images(
        args.input, 
        args.output, 
        max_workers=args.workers,
        max_iterations=args.iterations,
        batch_size=args.batch
    )
    total_time = time.time() - start_time
    
    print("\nSummary:")
    print(f"Total images processed: {summary_df['total_images'].values[0]}")
    print(f"Average original probability: {summary_df['avg_original_prob'].values[0]:.4f}")
    print(f"Average new probability: {summary_df['avg_new_prob'].values[0]:.4f}")
    print(f"Average difference: {summary_df['avg_difference'].values[0]:.4f}")
    print(f"Maximum difference: {summary_df['max_difference'].values[0]:.4f}")
    print(f"Success rate: {summary_df['success_rate'].values[0]:.2f}%")
    print(f"Average attack time per image: {summary_df['avg_attack_time'].values[0]:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")