import os
import numpy as np
import pandas as pd
import cv2

# Configuration
INPUT_DIR = "data/processed/NORMAL"  # Reads the .npy files we just created
OUTPUT_DIR = "data/results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "analysis_metrics.csv")

def analyze_single_image(file_path):
    """
    Loads a preprocessed image and calculates basic statistical metrics.
    Returns a dictionary of metrics.
    """
    try:
        # Load the .npy file (which contains the normalized float array)
        image_data = np.load(file_path)
        
        # Metric 1: Basic Intensity Stats (useful for detecting washed-out scans)
        mean_intensity = np.mean(image_data)
        std_intensity = np.std(image_data)
        
        # Metric 2: Edge Density (Texture Analysis)
        # We must convert back to 8-bit integer (0-255) for Canny Edge Detection
        img_uint8 = (image_data * 255).astype(np.uint8)
        edges = cv2.Canny(img_uint8, 100, 200)
        
        # Calculate the ratio of edge pixels to total pixels
        edge_density = np.count_nonzero(edges) / edges.size

        return {
            "filename": os.path.basename(file_path),
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "edge_density": edge_density
        }
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def run_analysis_pipeline(input_directory, output_csv_path):
    """
    Iterates through processed data, aggregates results, and saves to CSV.
    """
    # Create output directory if it doesn't exist
    if not os.path.dirname(output_csv_path):
        os.makedirs(os.path.dirname(output_csv_path))

    results = []
    
    print(f"Starting analysis on {input_directory}...")
    
    # Iterate over .npy files
    for filename in os.listdir(input_directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(input_directory, filename)
            
            metrics = analyze_single_image(file_path)
            
            if metrics:
                results.append(metrics)
    
    # Save to CSV using Pandas
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"Analysis complete. Results saved to {output_csv_path}")
        print(f"Total images analyzed: {len(df)}")
    else:
        print("No results found. Did you run preprocess.py first?")

if __name__ == "__main__":
    run_analysis_pipeline(INPUT_DIR, OUTPUT_FILE)