import numpy as np
from scipy.optimize import differential_evolution
import h5py
import os
import concurrent.futures
import argparse
import pandas as pd
import time
from tqdm import tqdm
import logging
import tensorflow as tf

#python hdf5_one_pixel_attack.py --normal_input /path/to/normal/hdf5s/ 
# --cancer_input /path/to/cancer/hdf5s/ --output /path/to/output/ 
# --model_path your_model.hdf5 --normal_dataset data --cancer_dataset data

# python hdf5_one_pixel_attack.py --normal_input /path/to/normal/hdf5s/
# --cancer_input /path/to/cancer/hdf5s/ --output /path/to/output/
# --normal_dataset images --cancer_dataset images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pixel_attack.log"),
        logging.StreamHandler()
    ]
)

MODEL_PATH = "0.29452_f1max_0.14705_f1_0.78622_loss_0_epoch_model.h5"  # Default path to the H5 model file
PRESET_COLORS = [[0, 0, 0], [255, 255, 255], [255, 255, 0]]  # based on research
model = None  # Will be loaded globally

def load_model(model_path):
    """Load the H5 model file"""
    global model
    try:
        model = tf.keras.models.load_model(model_path)
        logging.info(f"Successfully loaded model from {model_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return False

def preprocess_data(data_array):
    """Preprocess data array for model input"""
    try:
        # Ensure data is float and normalized (0-1)
        if data_array.dtype != np.float32 and data_array.dtype != np.float64:
            data_array = data_array.astype(np.float32)
            if data_array.max() > 1.0:
                data_array = data_array / 255.0
        
        # Reshape if needed (assuming model expects 64x64 images)
        if data_array.shape[-3:-1] != (64, 64):
            # Resize to 64x64
            from skimage.transform import resize
            data_array = resize(data_array, (64, 64, 3), anti_aliasing=True)
        
        # Add batch dimension if not present
        if len(data_array.shape) == 3:
            data_array = np.expand_dims(data_array, axis=0)
        
        return data_array
    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}")
        return None

def call_model(data_array):
    """Use the loaded model to get predictions"""
    try:
        # Ensure model is loaded
        if model is None:
            logging.error("Model not loaded. Call load_model first.")
            return {"predictions": [{"probability": 0}]}
        
        # Preprocess data
        preprocessed = preprocess_data(data_array)
        if preprocessed is None:
            return {"predictions": [{"probability": 0}]}
        
        # Get prediction
        predictions = model.predict(preprocessed, verbose=0)
        
        # Format output to match the original API response format
        probability = float(predictions[0][0])  # Assuming binary classification with cancer probability at index 0
        
        return {"predictions": [{"probability": probability}]}
    except Exception as e:
        logging.error(f"Error in call_model: {e}")
        return {"predictions": [{"probability": 0}]}

def one_pixel_attack(data, preset_colors, max_iter=100, patience=10):
    """Perform one-pixel attack with early stopping"""
    best_score = float('inf')
    patience_counter = 0
    
    def perturbation(params):
        nonlocal best_score, patience_counter
        
        data_copy = data.copy()
        x, y, color_idx = int(params[0]), int(params[1]), int(params[2])
        
        # Handle edge cases
        x = min(max(x, 0), data_copy.shape[1] - 1)
        y = min(max(y, 0), data_copy.shape[0] - 1)
        
        r, g, b = preset_colors[color_idx % len(preset_colors)]
        data_copy[y, x] = [r/255.0, g/255.0, b/255.0]  # Normalize color values
        
        # Access the correct key path in the response
        response = call_model(data_copy)
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
    
    # Return pixel info in RGB
    return [x, y, r/255.0, g/255.0, b/255.0]

def produce_altered_data(data, pixel):
    """Create a new data array with the pixel attack applied"""
    altered_data = data.copy()
    x, y, r, g, b = pixel
    
    # Ensure coordinates are within bounds
    x = min(max(int(x), 0), altered_data.shape[1] - 1)
    y = min(max(int(y), 0), altered_data.shape[0] - 1)
    
    altered_data[y, x] = [r, g, b]
    return altered_data

def process_single_hdf5_dataset(file_path, dataset_name, original_dir, adversarial_dir, results_list, 
                               preset_colors=PRESET_COLORS, max_iter=100, data_type="normal"):
    """Process a single dataset from an HDF5 file for pixel attack"""
    try:
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Read data from HDF5
        with h5py.File(file_path, 'r') as hf:
            if dataset_name not in hf:
                logging.error(f"Dataset {dataset_name} not found in {file_path}")
                return
                
            data = hf[dataset_name][:]
        
        # Get original prediction
        original_pred = call_model(data)
        original_prob = original_pred.get("predictions", [{}])[0].get("probability", 0)
        
        # Run attack
        start_time = time.time()
        optimal_pixel = one_pixel_attack(data, preset_colors, max_iter=max_iter)
        attack_time = time.time() - start_time
        
        # Create altered data
        altered = produce_altered_data(data, optimal_pixel)
        
        # Get new prediction
        new_pred = call_model(altered)
        new_prob = new_pred.get("predictions", [{}])[0].get("probability", 0)
        
        # Create new HDF5 files for original and altered data
        original_output_path = os.path.join(original_dir, f"{filename}_{dataset_name}_original.h5")
        adversarial_output_path = os.path.join(adversarial_dir, f"{filename}_{dataset_name}_adversarial.h5")
        
        # Save original data
        with h5py.File(original_output_path, 'w') as hf:
            hf.create_dataset(dataset_name, data=data)
            
        # Save altered data
        with h5py.File(adversarial_output_path, 'w') as hf:
            hf.create_dataset(dataset_name, data=altered)
        
        # Store results
        x, y, r, g, b = optimal_pixel
        results_list.append({
            "filename": f"{filename}_{dataset_name}",
            "type": data_type,
            "original_probability": original_prob,
            "new_probability": new_prob,
            "difference": original_prob - new_prob,
            "pixel_x": x,
            "pixel_y": y,
            "pixel_r": r,
            "pixel_g": g,
            "pixel_b": b,
            "attack_time": attack_time
        })
        
        logging.info(f"Processed {data_type} dataset {filename}:{dataset_name} - Original: {original_prob:.4f}, New: {new_prob:.4f}, Diff: {original_prob - new_prob:.4f}")
        
    except Exception as e:
        logging.error(f"Error processing {file_path}:{dataset_name}: {e}")

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

def get_hdf5_datasets(file_path):
    """Get all dataset names from an HDF5 file"""
    datasets = []
    
    def collect_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
    
    with h5py.File(file_path, 'r') as hf:
        hf.visititems(collect_datasets)
    
    return datasets

def batch_process_hdf5(normal_input_dir, cancer_input_dir, output_dir, 
                      normal_limit=20000, cancer_limit=1000, 
                      max_workers=8, max_iterations=100,
                      normal_dataset='data', cancer_dataset='data'):
    """Process normal and cancer HDF5 files separately with specified limits"""
    
    # Setup directory structure
    dirs = setup_directories(output_dir)
    
    # Get normal HDF5 files
    normal_files = [os.path.join(normal_input_dir, f) for f in os.listdir(normal_input_dir) 
                  if f.lower().endswith('.h5') or f.lower().endswith('.hdf5')]
    normal_files = normal_files[:normal_limit]
    logging.info(f"Found {len(normal_files)} normal HDF5 files to process (limit: {normal_limit})")
    
    # Get cancer HDF5 files
    cancer_files = [os.path.join(cancer_input_dir, f) for f in os.listdir(cancer_input_dir) 
                  if f.lower().endswith('.h5') or f.lower().endswith('.hdf5')]
    cancer_files = cancer_files[:cancer_limit]
    logging.info(f"Found {len(cancer_files)} cancer HDF5 files to process (limit: {cancer_limit})")
    
    # Results container
    results = []
    
    # Process normal files in parallel
    logging.info("Processing normal HDF5 files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path in normal_files:
            # For each file, check if specified dataset exists
            try:
                with h5py.File(file_path, 'r') as hf:
                    if normal_dataset in hf:
                        future = executor.submit(
                            process_single_hdf5_dataset, 
                            file_path, 
                            normal_dataset,
                            dirs["normal_original"],
                            dirs["normal_adversarial"],
                            results, 
                            PRESET_COLORS, 
                            max_iterations,
                            "normal"
                        )
                        futures.append(future)
                    else:
                        # Try to find any datasets in the file
                        all_datasets = get_hdf5_datasets(file_path)
                        if all_datasets:
                            for ds in all_datasets:
                                # Check data shape to see if it looks like an image
                                if len(hf[ds].shape) >= 2:
                                    future = executor.submit(
                                        process_single_hdf5_dataset, 
                                        file_path, 
                                        ds,
                                        dirs["normal_original"],
                                        dirs["normal_adversarial"],
                                        results, 
                                        PRESET_COLORS, 
                                        max_iterations,
                                        "normal"
                                    )
                                    futures.append(future)
                                    break  # Just use the first valid dataset
                        else:
                            logging.warning(f"No datasets found in {file_path}")
            except Exception as e:
                logging.error(f"Error examining {file_path}: {e}")
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing normal HDF5 files"):
            pass
    
    # Process cancer files in parallel
    logging.info("Processing cancer HDF5 files...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_path in cancer_files:
            # For each file, check if specified dataset exists
            try:
                with h5py.File(file_path, 'r') as hf:
                    if cancer_dataset in hf:
                        future = executor.submit(
                            process_single_hdf5_dataset, 
                            file_path, 
                            cancer_dataset,
                            dirs["cancer_original"],
                            dirs["cancer_adversarial"],
                            results, 
                            PRESET_COLORS, 
                            max_iterations,
                            "cancer"
                        )
                        futures.append(future)
                    else:
                        # Try to find any datasets in the file
                        all_datasets = get_hdf5_datasets(file_path)
                        if all_datasets:
                            for ds in all_datasets:
                                # Check data shape to see if it looks like an image
                                if len(hf[ds].shape) >= 2:
                                    future = executor.submit(
                                        process_single_hdf5_dataset, 
                                        file_path, 
                                        ds,
                                        dirs["cancer_original"],
                                        dirs["cancer_adversarial"],
                                        results, 
                                        PRESET_COLORS, 
                                        max_iterations,
                                        "cancer"
                                    )
                                    futures.append(future)
                                    break  # Just use the first valid dataset
                        else:
                            logging.warning(f"No datasets found in {file_path}")
            except Exception as e:
                logging.error(f"Error examining {file_path}: {e}")
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing cancer HDF5 files"):
            pass
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    if not results_df.empty:
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
            "total_datasets": len(normal_results),
            "avg_original_prob": normal_results["original_probability"].mean(),
            "avg_new_prob": normal_results["new_probability"].mean(),
            "avg_difference": normal_results["difference"].mean(),
            "max_difference": normal_results["difference"].max(),
            "success_rate": (normal_results["difference"] > 0).mean() * 100,
            "avg_attack_time": normal_results["attack_time"].mean()
        }
        
        cancer_summary = {
            "type": "cancer",
            "total_datasets": len(cancer_results),
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
    else:
        logging.warning("No results were generated. Check input files and dataset names.")
    
    return results_df if not results_df.empty else pd.DataFrame(), summary_df if 'summary_df' in locals() else pd.DataFrame(), dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch One-Pixel Attack for HDF5 Files')
    parser.add_argument('--normal_input', type=str, required=True, help='Input directory containing normal HDF5 files')
    parser.add_argument('--cancer_input', type=str, required=True, help='Input directory containing cancer HDF5 files')
    parser.add_argument('--output', type=str, required=True, help='Base output directory for all results')
    parser.add_argument('--normal_limit', type=int, default=1, help='Number of normal HDF5 files to process')
    parser.add_argument('--cancer_limit', type=int, default=1, help='Number of cancer HDF5 files to process')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--iterations', type=int, default=100, help='Max iterations for differential evolution')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the H5 model file')
    parser.add_argument('--normal_dataset', type=str, default='data', help='Dataset name in normal HDF5 files')
    parser.add_argument('--cancer_dataset', type=str, default='data', help='Dataset name in cancer HDF5 files')
    
    args = parser.parse_args()
    
    # Load the model
    if not load_model(args.model_path):
        logging.error(f"Failed to load model from {args.model_path}. Exiting.")
        exit(1)
    
    start_time = time.time()
    results_df, summary_df, output_dirs = batch_process_hdf5(
        args.normal_input, 
        args.cancer_input,
        args.output, 
        normal_limit=args.normal_limit,
        cancer_limit=args.cancer_limit,
        max_workers=args.workers,
        max_iterations=args.iterations,
        normal_dataset=args.normal_dataset,
        cancer_dataset=args.cancer_dataset
    )
    total_time = time.time() - start_time
    
    # Print summary for each type if results exist
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            data_type = row['type']
            print(f"\n{data_type.capitalize()} Datasets Summary:")
            print(f"Total datasets processed: {row['total_datasets']}")
            print(f"Average original probability: {row['avg_original_prob']:.4f}")
            print(f"Average new probability: {row['avg_new_prob']:.4f}")
            print(f"Average difference: {row['avg_difference']:.4f}")
            print(f"Maximum difference: {row['max_difference']:.4f}")
            print(f"Success rate: {row['success_rate']:.2f}%")
            print(f"Average attack time per dataset: {row['avg_attack_time']:.2f} seconds")
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"\nFiles saved to:")
    for name, path in output_dirs.items():
        print(f"- {name}: {path}")
