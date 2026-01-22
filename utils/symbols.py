"""
Utility functions for symbol detection (checkmarks, ticks, crosses, etc.)

This module provides functions to detect and classify symbols commonly found
in invoices, particularly checkmarks used for selection indicators.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CheckmarkResult:
    """Result of checkmark/tick mark detection."""
    present: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    checkmark_type: Optional[str] = None
    confidence: float = 0.0
    associated_value: Optional[int] = None


@dataclass  
class SymbolDetectionResult:
    """Result of symbol detection and classification."""
    symbol_type: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float


# Symbol patterns for text-based detection
CHECKMARK_SYMBOLS = {
    'tick': ['✓', '✔', '√', '☑'],
    'cross': ['✗', '✘', 'X', 'x'],
    'circle_filled': ['☑'],
    'circle_empty': ['☐'],
    'handwritten_mark': None  # Detected by shape, not text
}


def detect_symbol_type(text: str) -> Optional[str]:
    """
    Detect if text is a symbol type (checkmark, tick, cross, etc.)
    
    Args:
        text: Extracted text from OCR
        
    Returns:
        Symbol type string or None
    """
    stripped = text.strip()
    
    if not stripped:
        return None
    
    # Check for known checkmark patterns
    for symbol_type, patterns in CHECKMARK_SYMBOLS.items():
        if patterns and stripped in patterns:
            return symbol_type
    
    # Check for single characters that might be handwritten marks
    if len(stripped) == 1 and stripped.isalpha():
        return 'handwritten_mark'
    
    return None


def _calculate_contour_properties(contour) -> Dict:
    """
    Calculate properties of a contour for shape analysis.
    
    Args:
        contour: OpenCV contour
        
    Returns:
        Dictionary with area, perimeter, circularity, solidity, etc.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # Circularity (1.0 = perfect circle)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # Solidity (ratio of contour area to convex hull area)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # Extent (ratio of contour area to bounding box area)
    bbox_area = w * h
    extent = area / bbox_area if bbox_area > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'solidity': solidity,
        'extent': extent,
        'aspect_ratio': aspect_ratio,
        'width': w,
        'height': h,
        'bbox': (x, y, x + w, y + h)
    }


def _classify_shape(props: Dict) -> Tuple[Optional[str], float]:
    """
    Classify contour shape based on properties.
    
    Args:
        props: Contour properties dictionary
        
    Returns:
        Tuple of (shape_type, confidence)
    """
    area = props['area']
    circularity = props['circularity']
    solidity = props['solidity']
    aspect_ratio = props['aspect_ratio']
    
    # Tick mark characteristics
    # - Elongated (width > height)
    # - Lower circularity (< 0.5)
    # - Moderate solidity (> 0.3)
    is_tick = (
        aspect_ratio > 0.3 and
        aspect_ratio < 3.0 and
        solidity > 0.3 and
        circularity < 0.5
    )
    
    # Cross mark characteristics
    # - Roughly square
    # - Lower circularity
    is_cross = (
        aspect_ratio > 0.4 and
        aspect_ratio < 2.5 and
        circularity < 0.4
    )
    
    # Circle/ballot box characteristics
    # - High circularity (> 0.6)
    # - Near-square aspect ratio
    is_circle = (
        circularity > 0.6 and
        0.5 < aspect_ratio < 2.0
    )
    
    # Determine type and confidence
    if is_tick:
        return 'tick', 0.85
    elif is_cross:
        return 'cross', 0.80
    elif is_circle:
        return 'circle', 0.75
    else:
        # Default classification based on size
        if area < 1000:
            return 'small_mark', 0.65
        else:
            return 'unknown', 0.50


def detect_checkmarks(image: np.ndarray,
                      region: Optional[Tuple[int, int, int, int]] = None,
                      min_area_ratio: float = 0.001,
                      max_area_ratio: float = 0.05) -> List[CheckmarkResult]:
    """
    Detect checkmarks/tick marks in an image using contour analysis.
    
    Checkmarks are commonly used in invoices to mark selected options,
    especially for HP (horse power) selection in tabular data.
    
    Args:
        image: Input image (BGR or grayscale)
        region: Optional region of interest (x1, y1, x2, y2)
        min_area_ratio: Minimum area as ratio of search region (default 0.001)
        max_area_ratio: Maximum area as ratio of search region (default 0.05)
        
    Returns:
        List of CheckmarkResult objects
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    image_shape = (h, w)
    
    # Use specified region or full image
    if region:
        x1, y1, x2, y2 = region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return [CheckmarkResult(present=False, confidence=0.9)]
        search_region = gray[y1:y2, x1:x2]
    else:
        search_region = gray
        x1, y1 = 0, 0
    
    gh, gw = search_region.shape[:2]
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(search_region, (3, 3), 0)
    
    # Apply adaptive threshold for better contrast detection
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    checkmarks = []
    min_area = gh * gw * min_area_ratio
    max_area = gh * gw * max_area_ratio
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if area < min_area or area > max_area:
            continue
        
        # Calculate contour properties
        props = _calculate_contour_properties(contour)
        
        # Classify shape
        checkmark_type, type_confidence = _classify_shape(props)
        
        # Adjust confidence based on solidity and circularity
        confidence = type_confidence
        if props['solidity'] > 0.5:
            confidence += 0.1
        if props['circularity'] < 0.3:
            confidence += 0.05
        
        # Calculate bounding box in original coordinates
        orig_bbox = (
            x1 + props['bbox'][0],
            y1 + props['bbox'][1],
            x1 + props['bbox'][2],
            y1 + props['bbox'][3]
        )
        
        checkmarks.append(CheckmarkResult(
            present=True,
            bbox=orig_bbox,
            checkmark_type=checkmark_type,
            confidence=min(confidence, 0.95)
        ))
    
    if not checkmarks:
        return [CheckmarkResult(present=False, confidence=0.9)]
    
    logger = __import__('logging').getLogger(__name__)
    logger.info(f"Detected {len(checkmarks)} checkmarks in region")
    return checkmarks


def detect_single_checkmark(image: np.ndarray,
                            region: Optional[Tuple[int, int, int, int]] = None) -> CheckmarkResult:
    """
    Detect if a single checkmark exists in the specified region.
    
    Simplified version for quick checks.
    
    Args:
        image: Input image
        region: Optional region of interest
        
    Returns:
        Single CheckmarkResult
    """
    results = detect_checkmarks(image, region)
    return results[0] if results else CheckmarkResult(present=False)


def extract_checkmark_regions(image: np.ndarray,
                               margin: int = 10) -> List[Tuple[int, int, int, int]]:
    """
    Extract regions around detected checkmarks.
    
    Useful for subsequent OCR or analysis near checkmarks.
    
    Args:
        image: Input image
        margin: Margin to add around each checkmark
        
    Returns:
        List of expanded bounding boxes
    """
    checkmarks = detect_checkmarks(image)
    
    regions = []
    h, w = image.shape[:2]
    
    for checkmark in checkmarks:
        if not checkmark.present or checkmark.bbox is None:
            continue
        
        x1, y1, x2, y2 = checkmark.bbox
        
        # Expand region with margin
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        regions.append((x1, y1, x2, y2))
    
    return regions


# Aliases for backward compatibility
checkmark_detection = detect_checkmarks
symbol_classification = detect_symbol_type

