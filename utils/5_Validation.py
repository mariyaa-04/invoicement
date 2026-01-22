"""
Validation and Quality Assurance Module.

This module provides semantic reasoning and validation for extracted fields,
including cross-validation, consistency checks, and confidence scoring.

Step 5 of the invoice processing pipeline.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool
    confidence: float
    message: str = ""
    suggestions: List[str] = field(default_factory=list)


@dataclass
class CrossFieldValidation:
    """Cross-field validation rules and results."""
    model_dealer_consistent: ValidationResult = None
    hp_in_expected_range: ValidationResult = None
    cost_reasonable: ValidationResult = None
    model_hp_consistent: ValidationResult = None


# Expected HP ranges by brand (tractor HP typically 20-75)
BRAND_HP_RANGES = {
    'mahindra': (20, 75),
    'sonalika': (30, 60),
    'swaraj': (25, 60),
    'john deere': (25, 75),
    'john': (25, 75),
    'farmtrac': (25, 50),
    'escorts': (25, 55),
    'tafe': (25, 50),
    'new holland': (25, 75),
    'new': (25, 75),
    'kubota': (20, 60),
    'eicher': (20, 60),
    'massey ferguson': (20, 75),
    'massey': (20, 75),
    'force motors': (20, 50),
    'force': (20, 50),
}

# Brand to dealer name mappings
BRAND_TO_DEALER = {
    'mahindra': 'Mahindra Tractors',
    'sonalika': 'Sonalika Tractors',
    'swaraj': 'Swaraj',
    'john deere': 'John Deere',
    'john': 'John Deere',
    'farmtrac': 'Farmtrac',
    'escorts': 'Escorts',
    'tafe': 'Tafe',
    'new holland': 'New Holland',
    'new': 'New Holland',
    'kubota': 'Kubota',
    'eicher': 'Eicher',
    'massey ferguson': 'Massey Ferguson',
    'massey': 'Massey Ferguson',
    'force motors': 'Force Motors',
    'force': 'Force Motors',
}


def validate_hp_range(hp_value: int, model_name: str = None) -> ValidationResult:
    """
    Validate if HP value is in expected range for tractors.
    
    Args:
        hp_value: HP value to validate
        model_name: Optional model name for brand-specific validation
        
    Returns:
        ValidationResult
    """
    if hp_value is None:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            message="HP value is missing"
        )
    
    # General tractor HP range
    general_min, general_max = 15, 100
    
    if not (general_min <= hp_value <= general_max):
        return ValidationResult(
            is_valid=False,
            confidence=0.95,
            message=f"HP value {hp_value} is outside typical tractor range ({general_min}-{general_max})",
            suggestions=[f"Verify HP value should be between {general_min} and {general_max}"]
        )
    
    # Brand-specific validation
    if model_name:
        model_lower = model_name.lower()
        
        for brand, (min_hp, max_hp) in BRAND_HP_RANGES.items():
            if brand in model_lower:
                if min_hp <= hp_value <= max_hp:
                    return ValidationResult(
                        is_valid=True,
                        confidence=0.9,
                        message=f"HP value {hp_value} is valid for {brand}"
                    )
                else:
                    return ValidationResult(
                        is_valid=False,
                        confidence=0.7,
                        message=f"HP value {hp_value} is outside expected range for {brand} ({min_hp}-{max_hp})",
                        suggestions=[f"Typical HP for {brand} models: {min_hp}-{max_hp}"]
                    )
    
    return ValidationResult(
        is_valid=True,
        confidence=0.85,
        message=f"HP value {hp_value} is within valid tractor range"
    )


def validate_cost_range(cost_value: int) -> ValidationResult:
    """
    Validate if cost value is reasonable for tractors.
    
    Tractor costs typically range from 3-15 lakhs (300,000 - 1,500,000 INR).
    
    Args:
        cost_value: Cost value to validate
        
    Returns:
        ValidationResult
    """
    if cost_value is None:
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            message="Cost value is missing"
        )
    
    # Typical tractor cost range (in INR)
    min_cost = 200000  # 2 lakhs
    max_cost = 2000000  # 20 lakhs
    
    if cost_value < min_cost:
        return ValidationResult(
            is_valid=False,
            confidence=0.8,
            message=f"Cost {cost_value} seems too low for a tractor",
            suggestions=[f"Typical tractor cost should be above {min_cost:,} INR"]
        )
    
    if cost_value > max_cost:
        return ValidationResult(
            is_valid=False,
            confidence=0.8,
            message=f"Cost {cost_value} seems too high for a tractor",
            suggestions=[f"Typical tractor cost should be below {max_cost:,} INR"]
        )
    
    return ValidationResult(
        is_valid=True,
        confidence=0.85,
        message=f"Cost {cost_value:,} is within reasonable tractor range"
    )


def validate_model_dealer_consistency(model_name: str, dealer_name: str) -> ValidationResult:
    """
    Validate if model brand is consistent with dealer name.
    
    Args:
        model_name: Extracted model name
        dealer_name: Extracted dealer name
        
    Returns:
        ValidationResult
    """
    if not model_name or not dealer_name:
        return ValidationResult(
            is_valid=True,
            confidence=0.5,
            message="Cannot validate - missing model or dealer name"
        )
    
    model_lower = model_name.lower()
    dealer_lower = dealer_name.lower()
    
    # Extract brand from model (first word typically)
    model_brand = model_lower.split()[0] if model_lower.split() else ""
    
    # Check if brand is mentioned in dealer name
    brand_in_dealer = any(
        brand in dealer_lower 
        for brand in [model_brand] + list(BRAND_TO_DEALER.keys())
    )
    
    if brand_in_dealer:
        return ValidationResult(
            is_valid=True,
            confidence=0.9,
            message="Model brand is consistent with dealer name"
        )
    
    # Check for partial matches
    if len(model_brand) > 3:
        for brand in BRAND_TO_DEALER.keys():
            if brand in model_brand or model_brand in brand:
                return ValidationResult(
                    is_valid=True,
                    confidence=0.75,
                    message=f"Partial brand match found: {model_brand} vs {brand}"
                )
    
    return ValidationResult(
        is_valid=False,
        confidence=0.6,
        message="Model brand may not match dealer name",
        suggestions=["Consider verifying dealer authorization for this brand"]
    )


def validate_model_hp_consistency(model_name: str, hp_value: int) -> ValidationResult:
    """
    Validate if HP value is consistent with model name.
    
    Args:
        model_name: Extracted model name
        hp_value: Extracted HP value
        
    Returns:
        ValidationResult
    """
    if not model_name or hp_value is None:
        return ValidationResult(
            is_valid=True,
            confidence=0.5,
            message="Cannot validate - missing model or HP"
        )
    
    # Some models encode HP in their name (e.g., "575 DI" might be ~47 HP)
    # This is a heuristic validation
    
    # Check for numeric patterns in model name
    numbers = re.findall(r'\d+', model_name)
    
    if numbers:
        # Some models have HP-like numbers in name
        for num in numbers:
            num_val = int(num)
            if 20 <= num_val <= 100 and abs(num_val - hp_value) > 20:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.6,
                    message=f"Model {model_name} contains number {num_val} but HP is {hp_value}",
                    suggestions=["Verify HP extraction accuracy"]
                )
    
    return ValidationResult(
        is_valid=True,
        confidence=0.8,
        message="Model and HP values are consistent"
    )


def cross_validate_fields(dealer_name: str = None,
                          model_name: str = None,
                          hp_value: int = None,
                          cost_value: int = None) -> CrossFieldValidation:
    """
    Perform cross-validation of extracted fields.
    
    Args:
        dealer_name: Extracted dealer name
        model_name: Extracted model name
        hp_value: Extracted HP value
        cost_value: Extracted cost value
        
    Returns:
        CrossFieldValidation with all validation results
    """
    return CrossFieldValidation(
        model_dealer_consistent=validate_model_dealer_consistency(model_name, dealer_name),
        hp_in_expected_range=validate_hp_range(hp_value, model_name),
        cost_reasonable=validate_cost_range(cost_value),
        model_hp_consistent=validate_model_hp_consistency(model_name, hp_value)
    )


def compute_document_confidence(field_confidences: Dict[str, float]) -> float:
    """
    Compute overall document confidence from field confidences.
    
    Args:
        field_confidences: Dictionary of field name -> confidence
        
    Returns:
        Overall confidence score (0.0 to 1.0)
    """
    if not field_confidences:
        return 0.0
    
    confidences = list(field_confidences.values())
    
    # Weighted average (can be adjusted)
    weights = {
        'dealer_name': 0.2,
        'model_name': 0.25,
        'horse_power': 0.2,
        'asset_cost': 0.2,
        'signature': 0.075,
        'stamp': 0.075
    }
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for field, confidence in field_confidences.items():
        weight = weights.get(field, 0.1)
        weighted_sum += confidence * weight
        total_weight += weight
    
    if total_weight > 0:
        return round(weighted_sum / total_weight, 2)
    else:
        return round(sum(confidences) / len(confidences), 2)


def apply_confidence_adjustments(base_confidence: float,
                                  validation_results: CrossFieldValidation) -> float:
    """
    Adjust confidence based on validation results.
    
    Args:
        base_confidence: Base confidence from extraction
        validation_results: Cross-validation results
        
    Returns:
        Adjusted confidence score
    """
    adjustment = 0.0
    
    # Reduce confidence for validation failures
    if validation_results.model_dealer_consistent and not validation_results.model_dealer_consistent.is_valid:
        adjustment -= 0.05 * (1 - validation_results.model_dealer_consistent.confidence)
    
    if validation_results.hp_in_expected_range and not validation_results.hp_in_expected_range.is_valid:
        adjustment -= 0.1 * (1 - validation_results.hp_in_expected_range.confidence)
    
    if validation_results.cost_reasonable and not validation_results.cost_reasonable.is_valid:
        adjustment -= 0.1 * (1 - validation_results.cost_reasonable.confidence)
    
    adjusted = base_confidence + adjustment
    
    # Clamp to valid range
    return max(0.0, min(1.0, adjusted))


# Inference functions for missing fields
def infer_dealer_from_model(model_name: str) -> Tuple[Optional[str], float]:
    """
    Infer dealer name from model name.
    
    Args:
        model_name: Extracted model name
        
    Returns:
        Tuple of (inferred_dealer, confidence)
    """
    if not model_name:
        return None, 0.0
    
    model_lower = model_name.lower()
    brand = model_lower.split()[0] if model_lower.split() else ""
    
    if brand in BRAND_TO_DEALER:
        inferred = BRAND_TO_DEALER[brand]
        # Lower confidence for inferred values
        return inferred, 0.4
    
    return None, 0.0


def infer_hp_from_model(model_name: str) -> Tuple[Optional[int], float]:
    """
    Infer HP value from model name (if encoded).
    
    Args:
        model_name: Extracted model name
        
    Returns:
        Tuple of (inferred_hp, confidence)
    """
    if not model_name:
        return None, 0.0
    
    # Some models encode HP in their name
    numbers = re.findall(r'\d+', model_name)
    
    for num in numbers:
        hp_candidate = int(num)
        if 15 <= hp_candidate <= 100:
            return hp_candidate, 0.3  # Low confidence for inference
    
    return None, 0.0

