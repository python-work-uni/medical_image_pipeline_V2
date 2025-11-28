import os
import cv2
import numpy as np

# Configuration Constants
# Standard size for many ML models (e.g., ResNet)
TARGET_SIZE = (224, 224) 
INPUT_DIR = "chest_xray_images/chest_xray/train/NORMAL" # Adjust based on where you unzipped the data
OUTPUT_DIR = "data/processed/NORMAL"

def preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Reads an image, resizes it, and normalizes pixel values.
    Returns the processed image array or None if the read fails.
    """
    # 1. Read the image
    # cv2.imread loads the image in BGR format (Blue-Green-Red)
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None

    # 2. Resize
    # Resizing ensures all inputs have the same dimensions for the model
    img_resized = cv2.resize(img, target_size)

    # 3. Normalize
    # Convert from integer (0-255) to float (0.0-1.0) for numerical stability
    img_normalized = img_resized.astype('float32') / 255.0
    
    return img_normalized

def process_dataset(input_directory, output_directory):
    """
    Iterates through the input directory, processes images, and saves them.
    """
    # Create output directory if it doesn't exist (Robustness)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    processed_count = 0
    
    # Iterate over files in the input directory
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_directory, filename)
            
            # Process the single image
            processed_img = preprocess_image(input_path)
            
            if processed_img is not None:
                # 4. Save the processed data
                # We save as .npy (NumPy binary) because we are saving 
                # float arrays (0.0-1.0), not standard image files anymore.
                save_filename = os.path.splitext(filename)[0] + '.npy'
                output_path = os.path.join(output_directory, save_filename)
                
                np.save(output_path, processed_img)
                processed_count += 1
                
                # Optional: Print progress every 10 images
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} images...")

    print(f"Done. Successfully processed {processed_count} images.")

if __name__ == "__main__":
    # This block only runs if you execute the script directly, not if imported
    print("Starting preprocessing pipeline...")
    process_dataset(INPUT_DIR, OUTPUT_DIR)