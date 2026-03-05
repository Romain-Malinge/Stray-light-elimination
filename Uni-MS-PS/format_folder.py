# Format a RAW or other image folder to a folder containing only PNGs named 01.png, 02.png, etc.
# Supports RAW files with optional gamma correction

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

try:
    import rawpy
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False

def apply_gamma_correction(img, gamma=2.2):
    """Apply gamma correction to an image."""
    # Normalize to [0, 1]
    if img.dtype == np.uint16:
        img_normalized = img.astype(np.float32) / 65535.0
    else:
        img_normalized = img.astype(np.float32) / 255.0
    
    # Apply inverse gamma (1/gamma) for decoding
    img_corrected = np.power(img_normalized, 1.0 / gamma)
    
    # Convert back to original bit depth
    if img.dtype == np.uint16:
        return (img_corrected * 65535.0).astype(np.uint16)
    else:
        return (img_corrected * 255.0).astype(np.uint8)

def read_image(file_path):
    """Read image file (RAW or standard format)."""
    ext = Path(file_path).suffix.lower()
    
    # Handle RAW formats
    if ext in ['.raw', '.cr2', '.nef', '.arw', '.dng', '.raf', '.rw2', '.orf'] and HAS_RAWPY:
        try:
            with rawpy.imread(file_path) as raw:
                # Use postprocess for standard demosaicing
                img = raw.postprocess()
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Warning: Failed to read RAW file {file_path}: {e}")
            return None
    
    # Handle standard image formats
    img = cv2.imread(file_path)
    return img

def format_folder(path_folder, path_folder_save, apply_gamma=False, gamma=2.2):
    """
    Format a folder of images to PNGs with optional gamma correction.
    
    Args:
        path_folder: Input folder path
        path_folder_save: Output folder path
        apply_gamma: Whether to apply gamma correction
        gamma: Gamma value (default 2.2)
    """
    if not os.path.exists(path_folder_save):
        os.makedirs(path_folder_save)
    
    files = sorted([f for f in os.listdir(path_folder) if not f.startswith('.')])
    
    for i, file in enumerate(files):
        file_path = os.path.join(path_folder, file)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        img = read_image(file_path)
        
        if img is None:
            print(f"Skipping {file}")
            continue
        
        # Apply gamma correction if requested
        if apply_gamma:
            img = apply_gamma_correction(img, gamma)
        
        output_name = "{:02d}.png".format(i + 1)
        output_path = os.path.join(path_folder_save, output_name)
        cv2.imwrite(output_path, img)
        print(f"Saved {output_name} from {file}")
    
    print(f"Processed {len(files)} images to {path_folder_save}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a folder of images (RAW or standard) to PNG format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output folder for PNG images"
    )
    parser.add_argument(
        "--gamma",
        action="store_true",
        help="Apply gamma correction"
    )
    parser.add_argument(
        "--gamma-value",
        type=float,
        default=2.2,
        help="Gamma value for correction (default: 2.2)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        exit(1)
    
    format_folder(args.input, args.output, apply_gamma=args.gamma, gamma=args.gamma_value)