"""
Utility functions for handwriting detection and stroke analysis.

This module provides functions to analyze image regions and determine
whether text appears handwritten vs printed, along with stroke width analysis.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class HandwritingAnalysisResult:
    """Result of handwriting analysis on an image region."""
    is_handwritten: bool
    confidence: float
    avg_stroke_width: float
    variance_score: float
    component_count: int


def analyze_stroke_width(region: np.ndarray) -> HandwritingAnalysisResult:
    """
    Analyze stroke width patterns to detect handwriting.
    
    Handwritten text typically has:
    - Irregular stroke widths
    - Varying line thickness
    - Connected components with varied density
    
    Printed text tends to have:
    - Consistent stroke widths
    - Uniform thickness
    - Regular component patterns
    
    Args:
        region: Image region to analyze (grayscale, 2D array)
        
    Returns:
        HandwritingAnalysisResult with analysis details
    """
    if region is None or region.size == 0:
        return HandwritingAnalysisResult(
            is_handwritten=False,
            confidence=0.5,
            avg_stroke_width=0.0,
            variance_score=0.0,
            component_count=0
        )
    
    h, w = region.shape
    
    # Ensure region is grayscale
    if len(region.shape) == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    
    # Apply binary threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    if num_labels <= 1:
        return HandwritingAnalysisResult(
            is_handwritten=False,
            confidence=0.5,
            avg_stroke_width=0.0,
            variance_score=0.0,
            component_count=0
        )
    
    # Analyze component sizes (excluding background)
    component_sizes = stats[1:, cv2.CC_STAT_AREA]
    
    if len(component_sizes) == 0:
        return HandwritingAnalysisResult(
            is_handwritten=False,
            confidence=0.5,
            avg_stroke_width=0.0,
            variance_score=0.0,
            component_count=0
        )
    
    # Calculate statistics
    size_variance = np.var(component_sizes)
    size_mean = np.mean(component_sizes)
    size_std = np.std(component_sizes)
    
    # Normalize variance by mean (coefficient of variation approach)
    if size_mean > 0:
        normalized_variance = size_variance / (size_mean ** 2)
    else:
        normalized_variance = 0.0
    
    # Calculate variance score (higher = more irregular = more likely handwritten)
    # Printed text has low variance (uniform strokes)
    # Handwriting has high variance (inconsistent strokes)
    variance_score = min(normalized_variance * 10, 1.0)
    
    # Estimate average stroke width (sqrt of mean area is rough approximation)
    avg_stroke_width = np.sqrt(size_mean) if size_mean > 0 else 0.0
    
    # Determine if handwritten based on variance
    # Threshold of 0.1 means 10% normalized variance indicates handwriting
    is_handwritten = normalized_variance > 0.1
    
    # Calculate confidence based on variance strength
    if normalized_variance > 0.3:
        confidence = 0.9  # Strong handwriting signal
    elif normalized_variance > 0.2:
        confidence = 0.75  # Moderate signal
    elif normalized_variance > 0.1:
        confidence = 0.6  # Weak signal
    else:
        confidence = 0.4  # Likely printed
    
    # Adjust confidence based on component count
    # Handwriting typically has more components than printed text
    component_count = len(component_sizes)
    if component_count > 10 and is_handwritten:
        confidence = min(confidence + 0.1, 0.95)
    elif component_count < 3 and not is_handwritten:
        confidence = min(confidence + 0.1, 0.95)
    
    return HandwritingAnalysisResult(
        is_handwritten=is_handwritten,
        confidence=confidence,
        avg_stroke_width=avg_stroke_width,
        variance_score=variance_score,
        component_count=component_count
    )


def is_handwritten_region(region: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Quick check if an image region appears handwritten.
    
    Args:
        region: Image region to analyze
        threshold: Confidence threshold (default 0.5)
        
    Returns:
        True if region appears handwritten
    """
    result = analyze_stroke_width(region)
    return result.is_handwritten and result.confidence >= threshold


def get_handwriting_confidence(region: np.ndarray) -> float:
    """
    Get confidence score for handwriting detection.
    
    Args:
        region: Image region to analyze
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    result = analyze_stroke_width(region)
    return result.confidence


def analyze_text_block(image: np.ndarray, 
                       bbox: Tuple[int, int, int, int]) -> HandwritingAnalysisResult:
    """
    Analyze a text block (defined by bounding box) for handwriting characteristics.
    
    Args:
        image: Full image (grayscale or BGR)
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        HandwritingAnalysisResult
    """
    x1, y1, x2, y2 = bbox
    
    # Ensure bbox is within image bounds
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return HandwritingAnalysisResult(
            is_handwritten=False,
            confidence=0.5,
            avg_stroke_width=0.0,
            variance_score=0.0,
            component_count=0
        )
    
    # Extract region
    region = image[y1:y2, x1:x2]
    
    return analyze_stroke_width(region)


def batch_analyze_regions(image: np.ndarray, 
                          bboxes: List[Tuple[int, int, int, int]],
                          threshold: float = 0.5) -> List[HandwritingAnalysisResult]:
    """
    Analyze multiple regions for handwriting.
    
    Args:
        image: Full image
        bboxes: List of bounding boxes
        threshold: Confidence threshold
        
    Returns:
        List of HandwritingAnalysisResult
    """
    results = []
    for bbox in bboxes:
        result = analyze_text_block(image, bbox)
        results.append(result)
    return results


# Aliases for backward compatibility
stroke_analysis = analyze_stroke_width
handwriting_check = is_handwritten_region

