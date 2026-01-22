"""
Utility functions for color-based analysis (stamp detection, red ink detection, etc.)

This module provides functions to detect regions with specific color characteristics,
particularly useful for stamp detection in invoices (red/brown ink patterns).
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ColorRegionResult:
    """Result of color region detection."""
    regions: List[Tuple[int, int, int, int]]
    confidence: float
    color_type: str


@dataclass
class StampDetectionResult:
    """Result of stamp detection analysis."""
    present: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 0.0
    color_type: Optional[str] = None
    is_circular: bool = False


# HSV color ranges for common stamp colors
COLOR_RANGES = {
    'red': {
        'lower1': (0, 50, 50),
        'upper1': (10, 255, 255),
        'lower2': (170, 50, 50),
        'upper2': (180, 255, 255)
    },
    'brown': {
        'lower1': (10, 30, 30),
        'upper1': (30, 255, 255),
        'lower2': None,
        'upper2': None
    },
    'orange': {
        'lower1': (0, 100, 100),
        'upper1': (20, 255, 255),
        'lower2': None,
        'upper2': None
    },
    'blue': {
        'lower1': (100, 50, 50),
        'upper1': (130, 255, 255),
        'lower2': None,
        'upper2': None
    },
    'green': {
        'lower1': (40, 50, 50),
        'upper1': (80, 255, 255),
        'lower2': None,
        'upper2': None
    },
    'purple': {
        'lower1': (120, 30, 30),
        'upper1': (160, 255, 255),
        'lower2': None,
        'upper2': None
    }
}


def create_color_mask(hsv: np.ndarray, color_name: str) -> np.ndarray:
    """
    Create a binary mask for a specific color in HSV space.
    
    Args:
        hsv: Image in HSV color space
        color_name: Name of color to detect ('red', 'brown', 'orange', etc.)
        
    Returns:
        Binary mask where white = pixels of specified color
    """
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color: {color_name}. Supported: {list(COLOR_RANGES.keys())}")
    
    ranges = COLOR_RANGES[color_name]
    
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    # Handle colors that wrap around HSV spectrum (red)
    if ranges['lower2'] is not None:
        # First range
        lower1 = np.array(ranges['lower1'])
        upper1 = np.array(ranges['upper1'])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Second range
        lower2 = np.array(ranges['lower2'])
        upper2 = np.array(ranges['upper2'])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        lower = np.array(ranges['lower1'])
        upper = np.array(ranges['upper1'])
        mask = cv2.inRange(hsv, lower, upper)
    
    return mask


def detect_color_regions(image: np.ndarray,
                         color_name: str = 'red',
                         min_area_ratio: float = 0.005,
                         max_area_ratio: float = 0.15,
                         apply_morphology: bool = True) -> ColorRegionResult:
    """
    Detect regions of a specific color in an image.
    
    Args:
        image: Input image (BGR format)
        color_name: Color to detect ('red', 'brown', 'orange', etc.)
        min_area_ratio: Minimum area as ratio of image
        max_area_ratio: Maximum area as ratio of image
        apply_morphology: Whether to apply morphological operations
        
    Returns:
        ColorRegionResult with detected regions
    """
    if len(image.shape) != 3:
        # Grayscale image, can't do color detection
        return ColorRegionResult(regions=[], confidence=0.5, color_type=color_name)
    
    h, w = image.shape[:2]
    image_area = h * w
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create color mask
    mask = create_color_mask(hsv, color_name)
    
    # Apply morphological operations to clean up
    if apply_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    min_area = image_area * min_area_ratio
    max_area = image_area * max_area_ratio
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area or area > max_area:
            continue
        
        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / ch if ch > 0 else 0
        
        # Filter by aspect ratio (stamps are roughly circular or square)
        if 0.3 < aspect_ratio < 3.0:
            regions.append((x, y, x + cw, y + ch))
    
    # Calculate confidence based on region quality
    confidence = 0.5
    if len(regions) > 0:
        # Higher confidence if regions have good aspect ratios
        good_regions = sum(1 for r in regions for x1, y1, x2, y2 in [r] 
                          if 0.5 < (x2-x1)/(y2-y1) < 2.0)
        if good_regions > 0:
            confidence = 0.7 + (good_regions / len(regions)) * 0.2
    
    logger = __import__('logging').getLogger(__name__)
    logger.info(f"Detected {len(regions)} {color_name} regions")
    
    return ColorRegionResult(
        regions=regions,
        confidence=confidence,
        color_type=color_name
    )


def detect_red_ink_regions(image: np.ndarray,
                           min_area_ratio: float = 0.005,
                           max_area_ratio: float = 0.15) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions with red ink (common for stamps).
    
    Args:
        image: Input image (BGR format)
        min_area_ratio: Minimum area as ratio of image
        max_area_ratio: Maximum area as ratio of image
        
    Returns:
        List of bounding boxes for red ink regions
    """
    result = detect_color_regions(
        image, 'red',
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )
    return result.regions


def detect_brown_ink_regions(image: np.ndarray,
                             min_area_ratio: float = 0.005,
                             max_area_ratio: float = 0.15) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions with brown ink (common for stamps).
    
    Args:
        image: Input image (BGR format)
        min_area_ratio: Minimum area as ratio of image
        max_area_ratio: Maximum area as ratio of image
        
    Returns:
        List of bounding boxes for brown ink regions
    """
    result = detect_color_regions(
        image, 'brown',
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio
    )
    return result.regions


def detect_stamp_regions(image: np.ndarray,
                         min_area_ratio: float = 0.005,
                         max_area_ratio: float = 0.15) -> List[StampDetectionResult]:
    """
    Detect stamp regions in an image.
    
    Stamps are typically:
    - Red or brown ink
    - Roughly circular or square
    - Moderate size
    
    Args:
        image: Input image (BGR format)
        min_area_ratio: Minimum area as ratio of image
        max_area_ratio: Maximum area as ratio of image
        
    Returns:
        List of StampDetectionResult objects
    """
    h, w = image.shape[:2]
    image_area = h * w
    
    # Detect red and brown regions
    red_result = detect_color_regions(image, 'red', min_area_ratio, max_area_ratio)
    brown_result = detect_color_regions(image, 'brown', min_area_ratio, max_area_ratio)
    
    all_regions = []
    
    # Process red regions
    for bbox in red_result.regions:
        all_regions.append(StampDetectionResult(
            present=True,
            bbox=bbox,
            confidence=0.8,
            color_type='red',
            is_circular=_is_circular_region(image, bbox)
        ))
    
    # Process brown regions
    for bbox in brown_result.regions:
        all_regions.append(StampDetectionResult(
            present=True,
            bbox=bbox,
            confidence=0.75,
            color_type='brown',
            is_circular=_is_circular_region(image, bbox)
        ))
    
    if not all_regions:
        return [StampDetectionResult(present=False, confidence=0.9)]
    
    logger = __import__('logging').getLogger(__name__)
    logger.info(f"Detected {len(all_regions)} potential stamp regions")
    
    return all_regions


def _is_circular_region(image: np.ndarray, 
                        bbox: Tuple[int, int, int, int]) -> bool:
    """
    Check if a region is approximately circular.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        True if region appears circular
    """
    x1, y1, x2, y2 = bbox
    
    # Extract region
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    else:
        gray = image[y1:y2, x1:x2]
    
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False
    
    # Analyze largest contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    
    if perimeter == 0:
        return False
    
    # Calculate circularity
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Circular regions have circularity close to 1.0
    return circularity > 0.6


def detect_any_stamp(image: np.ndarray) -> StampDetectionResult:
    """
    Quick check if any stamp is present in the image.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Single StampDetectionResult with best detection
    """
    results = detect_stamp_regions(image)
    
    if not results:
        return StampDetectionResult(present=False, confidence=0.9)
    
    # Return the best result
    best = max(results, key=lambda r: r.confidence if r.present else 0)
    return best


# Aliases for backward compatibility
red_ink_detection = detect_red_ink_regions
stamp_detection = detect_stamp_regions

