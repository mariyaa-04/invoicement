"""
Output Generation Module.

This module handles conversion of extracted fields to JSON format
and file output operations.

Step 6 of the invoice processing pipeline.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime

logger = logging.getLogger(__name__)


def fields_to_dict(doc_id: str,
                   page_number: int,
                   dealer_name: Dict = None,
                   model_name: Dict = None,
                   horse_power: Dict = None,
                   asset_cost: Dict = None,
                   signature: Dict = None,
                   stamp: Dict = None,
                   document_confidence: float = 0.0,
                   metadata: Dict = None) -> Dict:
    """
    Convert extracted fields to a dictionary for JSON serialization.
    
    Args:
        doc_id: Document identifier
        page_number: Page number
        dealer_name: Dealer name field result
        model_name: Model name field result
        horse_power: Horse power field result
        asset_cost: Asset cost field result
        signature: Signature detection result
        stamp: Stamp detection result
        document_confidence: Overall document confidence
        metadata: Optional additional metadata
        
    Returns:
        Dictionary ready for JSON serialization
    """
    output = {
        'doc_id': doc_id,
        'page_number': page_number,
        'dealer_name': dealer_name or {'value': None, 'confidence': 0.0},
        'model_name': model_name or {'value': None, 'confidence': 0.0},
        'horse_power': horse_power or {'value': None, 'confidence': 0.0},
        'asset_cost': asset_cost or {'value': None, 'confidence': 0.0},
        'signature': signature or {'present': False, 'confidence': 0.0},
        'stamp': stamp or {'present': False, 'confidence': 0.0},
        'document_confidence': document_confidence,
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    # Add metadata if provided
    if metadata:
        output['metadata'] = metadata
    
    return output


def field_result_to_dict(value, confidence, bbox=None, source=None) -> Dict:
    """
    Convert a FieldResult-like object to dictionary.
    
    Args:
        value: Field value
        confidence: Confidence score
        bbox: Optional bounding box
        source: Optional source ('text' or 'visual')
        
    Returns:
        Dictionary representation
    """
    result = {
        'value': value,
        'confidence': confidence
    }
    
    if bbox:
        result['bbox'] = list(bbox) if isinstance(bbox, tuple) else bbox
    
    if source:
        result['source'] = source
    
    return result


def signature_result_to_dict(present: bool, bbox=None, confidence: float = 0.0) -> Dict:
    """
    Convert a SignatureResult-like object to dictionary.
    
    Args:
        present: Whether signature is present
        bbox: Optional bounding box
        confidence: Confidence score
        
    Returns:
        Dictionary representation
    """
    result = {
        'present': present,
        'confidence': confidence
    }
    
    if bbox:
        result['bbox'] = list(bbox) if isinstance(bbox, tuple) else bbox
    
    return result


def stamp_result_to_dict(present: bool, bbox=None, confidence: float = 0.0) -> Dict:
    """
    Convert a StampResult-like object to dictionary.
    
    Args:
        present: Whether stamp is present
        bbox: Optional bounding box
        confidence: Confidence score
        
    Returns:
        Dictionary representation
    """
    result = {
        'present': present,
        'confidence': confidence
    }
    
    if bbox:
        result['bbox'] = list(bbox) if isinstance(bbox, tuple) else bbox
    
    return result


def to_json(data: Dict, filepath: str = None, indent: int = 2) -> str:
    """
    Convert data to JSON string or file.
    
    Args:
        data: Dictionary to serialize
        filepath: Optional file path to save JSON
        indent: JSON indentation level
        
    Returns:
        JSON string (and saves to file if filepath provided)
    """
    json_str = json.dumps(data, indent=indent, ensure_ascii=False)
    
    if filepath:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        logger.info(f"JSON output saved to {filepath}")
    
    return json_str


def save_extraction_result(result_dict: Dict,
                           output_dir: str,
                           doc_id: str,
                           page_number: int = 1) -> str:
    """
    Save extraction result to JSON file.
    
    Args:
        result_dict: Dictionary with extraction results
        output_dir: Output directory
        doc_id: Document identifier
        page_number: Page number
        
    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{doc_id}_page{page_number}_extracted.json"
    filepath = os.path.join(output_dir, filename)
    
    to_json(result_dict, filepath)
    
    return filepath


def create_batch_output(results: Dict[str, Dict],
                        output_dir: str = None) -> Dict[str, str]:
    """
    Save multiple extraction results and return file paths.
    
    Args:
        results: Dictionary of doc_id -> extraction result dict
        output_dir: Output directory (optional)
        
    Returns:
        Dictionary of doc_id -> file path
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    file_paths = {}
    
    for doc_id, result in results.items():
        if output_dir:
            filepath = save_extraction_result(
                result, output_dir, doc_id,
                result.get('page_number', 1)
            )
            file_paths[doc_id] = filepath
        else:
            file_paths[doc_id] = json.dumps(result, indent=2, ensure_ascii=False)
    
    return file_paths


def load_json_result(filepath: str) -> Dict:
    """
    Load extraction result from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with extraction results
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_currency(value: int, currency: str = 'INR') -> str:
    """
    Format currency value for display.
    
    Args:
        value: Numeric value
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    if value is None:
        return f"{currency} N/A"
    
    return f"{currency} {value:,}"


def format_percentage(confidence: float) -> str:
    """
    Format confidence as percentage.
    
    Args:
        confidence: Confidence score (0.0 to 1.0)
        
    Returns:
        Formatted percentage string
    """
    return f"{confidence * 100:.1f}%"


def generate_summary_report(results: Dict[str, Dict]) -> str:
    """
    Generate a summary report of extraction results.
    
    Args:
        results: Dictionary of doc_id -> extraction result dict
        
    Returns:
        Formatted summary report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("INVOICE EXTRACTION SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append(f"Total Documents: {len(results)}")
    lines.append("")
    
    total_confidence = 0
    dealer_found = 0
    model_found = 0
    hp_found = 0
    cost_found = 0
    signature_found = 0
    stamp_found = 0
    
    for doc_id, result in results.items():
        doc_conf = result.get('document_confidence', 0)
        total_confidence += doc_conf
        
        if result.get('dealer_name', {}).get('value'):
            dealer_found += 1
        if result.get('model_name', {}).get('value'):
            model_found += 1
        if result.get('horse_power', {}).get('value'):
            hp_found += 1
        if result.get('asset_cost', {}).get('value'):
            cost_found += 1
        if result.get('signature', {}).get('present'):
            signature_found += 1
        if result.get('stamp', {}).get('present'):
            stamp_found += 1
        
        lines.append(f"\nDocument: {doc_id}")
        lines.append(f"  Dealer: {result.get('dealer_name', {}).get('value', 'N/A')}")
        lines.append(f"  Model: {result.get('model_name', {}).get('value', 'N/A')}")
        lines.append(f"  HP: {result.get('horse_power', {}).get('value', 'N/A')}")
        lines.append(f"  Cost: {result.get('asset_cost', {}).get('value', 'N/A')}")
        lines.append(f"  Signature: {'Yes' if result.get('signature', {}).get('present') else 'No'}")
        lines.append(f"  Stamp: {'Yes' if result.get('stamp', {}).get('present') else 'No'}")
        lines.append(f"  Confidence: {format_percentage(doc_conf)}")
    
    lines.append("\n" + "=" * 60)
    lines.append("AGGREGATE STATISTICS")
    lines.append("=" * 60)
    lines.append(f"Average Confidence: {format_percentage(total_confidence / max(len(results), 1))}")
    lines.append(f"Dealer Found: {dealer_found}/{len(results)} ({format_percentage(dealer_found / max(len(results), 1))})")
    lines.append(f"Model Found: {model_found}/{len(results)} ({format_percentage(model_found / max(len(results), 1))})")
    lines.append(f"HP Found: {hp_found}/{len(results)} ({format_percentage(hp_found / max(len(results), 1))})")
    lines.append(f"Cost Found: {cost_found}/{len(results)} ({format_percentage(cost_found / max(len(results), 1))})")
    lines.append(f"Signature Present: {signature_found}/{len(results)} ({format_percentage(signature_found / max(len(results), 1))})")
    lines.append(f"Stamp Present: {stamp_found}/{len(results)} ({format_percentage(stamp_found / max(len(results), 1))})")
    
    return "\n".join(lines)

