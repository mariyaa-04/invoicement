"""
Utility functions for the invoice processing pipeline.

This module re-exports utilities from submodules for convenient access.
"""

# Handwriting analysis
from .handwriting import (
    analyze_stroke_width,
    analyze_text_block,
    batch_analyze_regions,
    is_handwritten_region,
    get_handwriting_confidence,
    HandwritingAnalysisResult
)

# Symbol detection
from .symbols import (
    detect_checkmarks,
    detect_symbol_type,
    detect_single_checkmark,
    extract_checkmark_regions,
    CheckmarkResult,
    SymbolDetectionResult
)

# Color analysis
from .color_analysis import (
    detect_red_ink_regions,
    detect_brown_ink_regions,
    detect_stamp_regions,
    detect_any_stamp,
    detect_color_regions,
    ColorRegionResult,
    StampDetectionResult
)

__all__ = [
    # Handwriting
    'analyze_stroke_width',
    'analyze_text_block',
    'batch_analyze_regions',
    'is_handwritten_region',
    'get_handwriting_confidence',
    'HandwritingAnalysisResult',
    
    # Symbols
    'detect_checkmarks',
    'detect_symbol_type',
    'detect_single_checkmark',
    'extract_checkmark_regions',
    'CheckmarkResult',
    'SymbolDetectionResult',
    
    # Color
    'detect_red_ink_regions',
    'detect_brown_ink_regions',
    'detect_stamp_regions',
    'detect_any_stamp',
    'detect_color_regions',
    'ColorRegionResult',
    'StampDetectionResult'
]

