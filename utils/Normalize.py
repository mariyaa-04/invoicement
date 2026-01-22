"""
STEP 1: Image Normalization

Purpose of Step 1:
- Preprocess invoice images to a standardized format
- Handle multiple page invoices
- Perform quality checks
- Prepare images for consistent downstream processing

Input: Raw invoice images (PNG format)
Output: Normalized images and metadata for Step 2

Author: Document AI System
Version: 2.0
"""

import os
import cv2
import numpy as np
from PIL import Image
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import tempfile
import shutil

def normalize_image(image, target_width=800, target_height=600):
    """
    Normalize an image to consistent resolution and format for OCR.
    Keep it simple: grayscale, light noise reduction, optional mild contrast.

    Args:
        image (PIL.Image or numpy.ndarray): Input image
        target_width (int): Target width in pixels
        target_height (int): Target height in pixels

    Returns:
        numpy.ndarray: Normalized image (grayscale, lightly processed)
    """
    if isinstance(image, Image.Image):
        # Convert PIL to numpy array
        img_array = np.array(image)
    else:
        img_array = image

    # Convert to grayscale if not already
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Light noise reduction using small Gaussian blur
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

    # Optional: mild contrast normalization (CLAHE for local contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)

    # Resize to target dimensions
    img_array = cv2.resize(img_array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return img_array

def correct_skew(image):
    """
    Correct skew in the image using Hough transform.

    Args:
        image (numpy.ndarray): Grayscale image

    Returns:
        numpy.ndarray: Skew-corrected image
    """
    # Threshold the image to get binary image
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # No contours found, return original

    # Find the largest contour (assuming it's the main text area)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    # Adjust angle if necessary (minAreaRect gives angle between -90 and 0)
    if angle < -45:
        angle = 90 + angle

    # Rotate the image to correct skew
    if abs(angle) > 0.5:  # Only rotate if skew is significant
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image

def quality_check(image, min_resolution=(100, 100)):
    """
    Perform basic quality check on the image.

    Args:
        image (numpy.ndarray): Image to check
        min_resolution (tuple): Minimum acceptable resolution (width, height)

    Returns:
        bool: True if image passes quality check, False otherwise
    """
    # Check if image is blank (all pixels are the same or very low variance)
    if image.size == 0:
        logging.warning("Image is empty")
        return False

    # Check resolution
    h, w = image.shape[:2]
    if w < min_resolution[0] or h < min_resolution[1]:
        logging.warning(f"Image resolution too low: {w}x{h}")
        return False

    # Check if image is corrupted (e.g., all black or all white)
    unique_pixels = np.unique(image)
    if len(unique_pixels) <= 1:
        logging.warning("Image appears to be blank or corrupted")
        return False

    # Check for excessive noise (high variance might indicate corruption)
    variance = np.var(image)
    if variance < 10:  # Very low variance might indicate blank image
        logging.warning("Image has very low variance, possibly blank")
        return False

    return True

def process_png(file_path):
    """
    Load and prepare PNG image file.

    Args:
        file_path (str): Path to PNG file

    Returns:
        PIL.Image: Loaded image
    """
    try:
        image = Image.open(file_path)
        return image
    except Exception as e:
        raise ValueError(f"Error loading PNG {file_path}: {str(e)}")

def normalize_invoice_pages(png_paths, output_dir=None, target_width=800, target_height=600):
    """
    Main function to normalize invoice pages from PNG images.

    Args:
        png_paths (list): List of paths to PNG files (each representing one page)
        output_dir (str, optional): Directory to save normalized images. If None, images are not saved.
        target_width (int): Target width for normalization
        target_height (int): Target height for normalization

    Returns:
        list: List of dicts with normalized images and metadata
              Each dict: {'image': numpy.ndarray, 'page': int, 'original_filename': str}
    """
    normalized_pages = []

    for page_num, file_path in enumerate(png_paths):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext != '.png':
            raise ValueError(f"Only PNG files are supported: {file_path}")

        original_filename = os.path.basename(file_path)

        # Load and normalize the image
        page_image = process_png(file_path)
        normalized_img = normalize_image(page_image, target_width, target_height)

        # Save if output_dir is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{os.path.splitext(original_filename)[0]}_normalized.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, normalized_img)

        # Create metadata dict
        page_data = {
            'image': normalized_img,
            'page': page_num + 1,
            'original_filename': original_filename
        }
        normalized_pages.append(page_data)

    return normalized_pages

def identify_png_files(directory):
    """
    Identify all PNG files in a directory.

    Args:
        directory (str): Path to directory containing PNG files

    Returns:
        list: List of full paths to PNG files
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    png_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png')]
    return png_files

def group_invoice_pages(png_files):
    """
    Group PNG files that belong to the same invoice based on filename pattern.
    Tolerant parsing to handle various patterns like:
    - 172427893_3_pg11.png (invoice_id _ something _ pg page_num)
    - 172561841_pg1.png (invoice_id _ pg page_num)
    - 172808806_pg14.png (invoice_id _ pg page_num, no something)

    Args:
        png_files (list): List of PNG file paths

    Returns:
        dict: Dictionary with invoice_id as key, list of (file_path, page_num) tuples as value
    """
    invoice_groups = {}

    for file_path in png_files:
        filename = os.path.basename(file_path)
        # Remove .png extension
        filename_no_ext = filename.replace('.png', '')

        # Try to match various patterns
        # Pattern 1: invoice_id _ something _ pg page_num
        match1 = re.match(r'^(\d+)_(\d+)_pg(\d+)$', filename_no_ext)
        if match1:
            invoice_id = match1.group(1)
            page_num = int(match1.group(3))
        else:
            # Pattern 2: invoice_id _ pg page_num (no something)
            match2 = re.match(r'^(\d+)_pg(\d+)$', filename_no_ext)
            if match2:
                invoice_id = match2.group(1)
                page_num = int(match2.group(2))
            else:
                # Fallback: assume single-page invoice, use filename as invoice_id, page 1
                invoice_id = filename_no_ext
                page_num = 1

        if invoice_id not in invoice_groups:
            invoice_groups[invoice_id] = []
        invoice_groups[invoice_id].append((file_path, page_num))

    # Sort pages within each invoice by page number
    for invoice_id in invoice_groups:
        invoice_groups[invoice_id].sort(key=lambda x: x[1])

    return invoice_groups

def assign_page_numbers(invoice_groups):
    """
    Assign sequential page numbers to invoice pages (in case the extracted page numbers are not sequential).

    Args:
        invoice_groups (dict): Dictionary from group_invoice_pages

    Returns:
        dict: Dictionary with invoice_id as key, list of (file_path, assigned_page_num) tuples as value
    """
    assigned_groups = {}

    for invoice_id, pages in invoice_groups.items():
        # Sort by original page number and assign sequential page numbers
        sorted_pages = sorted(pages, key=lambda x: x[1])
        assigned_pages = [(file_path, idx + 1) for idx, (file_path, _) in enumerate(sorted_pages)]
        assigned_groups[invoice_id] = assigned_pages

    return assigned_groups

def normalize_invoice_directory(input_dir, output_dir=None, target_width=800, target_height=600):
    """
    Normalize all PNG files in a directory (useful for multi-page invoices).

    Args:
        input_dir (str): Directory containing PNG files
        output_dir (str, optional): Directory to save normalized images
        target_width (int): Target width for normalization
        target_height (int): Target height for normalization

    Returns:
        list: List of dicts with normalized images and metadata
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    png_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    png_files.sort()  # Sort to maintain page order

    return normalize_invoice_pages(png_files, output_dir, target_width, target_height)

def process_step1(input_dir, output_dir=None, target_width=800, target_height=600, min_resolution=(100, 100)):
    """
    Main function to process STEP 1: Document Input for the dataset.
    Reads all PNG files from input_dir, groups by invoice ID, sorts by page number,
    performs quality checks, normalizes images, and prepares output for OCR.

    Args:
        input_dir (str): Directory containing PNG files (e.g., train_data_idfc/train)
        output_dir (str, optional): Directory to save normalized images for debugging. If None, images are kept in memory.
        target_width (int): Target width for normalization
        target_height (int): Target height for normalization
        min_resolution (tuple): Minimum acceptable resolution (width, height) for quality check

    Returns:
        dict: Dictionary with invoice_id as key, list of page dicts as value.
              Each page dict: {'invoice_id': str, 'page_number': int, 'normalized_image': numpy.ndarray}
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Read All PNG Files
    png_files = identify_png_files(input_dir)
    logging.info(f"Found {len(png_files)} PNG files in {input_dir}")

    # 2. Parse Filename Metadata and Group by Invoice ID
    invoice_groups = group_invoice_pages(png_files)
    logging.info(f"Grouped into {len(invoice_groups)} invoices")

    # Assign sequential page numbers starting from 1 for each invoice
    assigned_groups = assign_page_numbers(invoice_groups)

    # 3. Prepare Standard Output
    processed_invoices = {}

    for invoice_id, pages in assigned_groups.items():
        processed_pages = []

        for file_path, assigned_page_num in pages:
            try:
                # 4. Load Each Image
                page_image = process_png(file_path)

                # Convert to numpy array for quality check
                img_array = np.array(page_image) if isinstance(page_image, Image.Image) else page_image
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

                # 5. Basic Quality Check
                if not quality_check(img_array, min_resolution):
                    logging.warning(f"Skipping {file_path} due to quality check failure")
                    continue

                # 6. Image Normalization
                normalized_img = normalize_image(page_image, target_width, target_height)

                # 7. Prepare Standard Output
                page_data = {
                    'invoice_id': invoice_id,
                    'page_number': assigned_page_num,
                    'normalized_image': normalized_img
                }
                processed_pages.append(page_data)

                # Save for debugging if output_dir is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f"{invoice_id}_pg{assigned_page_num}_normalized.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, normalized_img)

            except Exception as e:
                logging.error(f"Error processing {file_path}: {str(e)}")
                continue

        if processed_pages:
            # Sort by page number
            processed_pages.sort(key=lambda x: x['page_number'])
            processed_invoices[invoice_id] = processed_pages

    logging.info(f"Successfully processed {len(processed_invoices)} invoices with {sum(len(pages) for pages in processed_invoices.values())} pages")
    return processed_invoices

# Example usage
if __name__ == "__main__":
    # Test with sample invoice
    sample_path = "data/images/sample_invoice.png"
    if os.path.exists(sample_path):
        result = normalize_invoice_pages([sample_path], output_dir="normalized_output")
        print(f"Normalized {len(result)} pages")
        for page in result:
            print(f"Page {page['page']}: {page['original_filename']}, shape: {page['image'].shape}")
    else:
        print("Sample invoice not found. Run create_sample.py first.")

    # Test STEP 1 Process
    train_dir = "train_data_idfc/train"
    if os.path.exists(train_dir):
        print("\n--- STEP 1 Process Test ---")
        # Process a small subset for testing (first 10 files)
        png_files = identify_png_files(train_dir)[:10]
        print(f"Testing with {len(png_files)} PNG files")

        # Test grouping
        invoice_groups = group_invoice_pages(png_files)
        print(f"Grouped into {len(invoice_groups)} invoices")

        # Show sample groupings
        sample_invoices = list(invoice_groups.keys())[:3]  # Show first 3
        for invoice_id in sample_invoices:
            pages = invoice_groups[invoice_id]
            print(f"Invoice {invoice_id}: {len(pages)} pages")
            for file_path, page_num in pages[:2]:  # Show first 2 pages
                filename = os.path.basename(file_path)
                print(f"  Page {page_num}: {filename}")

        # Test full process_step1 with small subset
        print("\n--- Testing process_step1 with subset ---")
        # Create a temporary directory with subset
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_path in png_files:
                import shutil
                shutil.copy(file_path, temp_dir)

            result = process_step1(temp_dir, output_dir="normalized_output")
            print(f"Processed {len(result)} invoices")
            total_pages = sum(len(pages) for pages in result.values())
            print(f"Total pages processed: {total_pages}")

            # Show sample result
            for invoice_id, pages in list(result.items())[:2]:
                print(f"Invoice {invoice_id}: {len(pages)} pages")
                for page in pages[:2]:
                    print(f"  Page {page['page_number']}: shape {page['normalized_image'].shape}")
    else:
        print(f"Training directory {train_dir} not found.")
