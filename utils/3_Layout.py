"""
STEP 3: Layout Understanding (Structural Interpretation)

Purpose of Step 3:
- Convert raw OCR text into spatial structure
- Group text blocks by relative vertical position
- Identify regions: header (top), body (middle), footer (bottom)
- Preserve spatial context without interpretation

Why This Is Critical:
- Step 3 does NOT know what a dealer name is
- Step 3 does NOT extract fields
- Step 3 only answers: "Which text is near the top/middle/bottom?"

Author: Document AI System
Version: 2.0
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OCRBlock:
    """
    OCR text element with position and confidence (from Step 2)
    """
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class PageLayout:
    """
    Simple spatial layout for a single page
    """
    doc_id: str
    page_number: int
    header_blocks: List[OCRBlock] = field(default_factory=list)
    body_blocks: List[OCRBlock] = field(default_factory=list)
    footer_blocks: List[OCRBlock] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_id': self.doc_id,
            'page_number': self.page_number,
            'header_blocks': [{'text': b.text, 'bbox': b.bbox, 'confidence': b.confidence} for b in self.header_blocks],
            'body_blocks': [{'text': b.text, 'bbox': b.bbox, 'confidence': b.confidence} for b in self.body_blocks],
            'footer_blocks': [{'text': b.text, 'bbox': b.bbox, 'confidence': b.confidence} for b in self.footer_blocks]
        }

    def to_json(self, filepath: str = None) -> str:
        """Export to JSON string or file"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Layout data saved to {filepath}")
        return json_str


class LayoutAnalyzer:
    """
    Simple spatial analyzer - groups blocks by relative vertical position
    """

    # Configuration parameters - relative ratios work across different page sizes
    HEADER_RATIO = 0.35  # top 35% of page is header
    FOOTER_RATIO = 0.25  # bottom 25% of page is footer
    # Body is the middle: 35% to 75%

    def __init__(self, page_height: int = 600):
        """
        Initialize layout analyzer

        Args:
            page_height: Expected page height in pixels (default works for normalized images)
        """
        self.page_height = page_height

    def analyze_layout(self, ocr_blocks: List[OCRBlock]) -> PageLayout:
        """
        Group OCR blocks into spatial regions based on relative vertical position

        Args:
            ocr_blocks: Raw OCR blocks from Step 2

        Returns:
            PageLayout with blocks grouped by region
        """
        logger.info(f"Starting spatial layout analysis for {len(ocr_blocks)} OCR blocks")

        # Calculate thresholds based on relative position
        header_threshold = int(self.page_height * self.HEADER_RATIO)
        footer_threshold = int(self.page_height * (1 - self.FOOTER_RATIO))

        logger.info(f"Page height: {self.page_height}, Header threshold: {header_threshold}, Footer threshold: {footer_threshold}")

        # Initialize regions
        header_blocks = []
        body_blocks = []
        footer_blocks = []

        # Group blocks by relative vertical position
        for block in ocr_blocks:
            center_y = block.center_y

            if center_y <= header_threshold:
                header_blocks.append(block)
            elif center_y >= footer_threshold:
                footer_blocks.append(block)
            else:
                body_blocks.append(block)

        # Sort blocks within each region by vertical position (top to bottom)
        header_blocks.sort(key=lambda b: b.center_y)
        body_blocks.sort(key=lambda b: b.center_y)
        footer_blocks.sort(key=lambda b: b.center_y)

        # Create page layout
        page_layout = PageLayout(
            doc_id='unknown',  # Will be set by caller
            page_number=1,     # Will be set by caller
            header_blocks=header_blocks,
            body_blocks=body_blocks,
            footer_blocks=footer_blocks
        )

        logger.info(f"Layout analysis complete - Header: {len(header_blocks)}, Body: {len(body_blocks)}, Footer: {len(footer_blocks)}")
        return page_layout


def load_ocr_output(ocr_dir: str, limit: int = 100) -> Dict[str, Dict]:
    """
    Load OCR JSON files from directory (randomly selected up to limit)

    Args:
        ocr_dir: Directory containing OCR JSON files
        limit: Maximum number of files to load (default: 100)

    Returns:
        Dictionary with doc_id as key, OCR data as value
    """
    ocr_data = {}

    if not os.path.exists(ocr_dir):
        logger.warning(f"OCR directory {ocr_dir} does not exist")
        return ocr_data

    # Get all JSON files
    all_files = [f for f in os.listdir(ocr_dir) if f.endswith('.json')]
    
    # Randomly select files if limit is set
    if limit and limit < len(all_files):
        selected_files = random.sample(all_files, limit)
    else:
        selected_files = all_files

    for filename in selected_files:
        filepath = os.path.join(ocr_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            doc_id = data.get('doc_id', filename.replace('_ocr_output.json', ''))
            ocr_data[doc_id] = data

            logger.info(f"Loaded OCR data for {doc_id}")

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded {len(ocr_data)} OCR files (limit: {limit})")
    return ocr_data


def process_step3(ocr_data: Dict[str, Dict], output_dir: str = None,
                  limit: int = 100) -> Dict[str, PageLayout]:
    """
    STEP 3: Layout Understanding - Main processing function

    Takes OCR data from Step 2 and produces simple spatial grouping.

    INPUT (from Step 2):
    {
        "doc_id": "172610467",
        "page_number": 1,
        "ocr_blocks": [
            {"text": "IDFC FIRST BANK", "bbox": [x1, y1, x2, y2], "confidence": 0.82},
            ...
        ]
    }

    OUTPUT:
    {
        "doc_id": "172610467",
        "page_number": 1,
        "header_blocks": [...],
        "body_blocks": [...],
        "footer_blocks": [...]
    }

    Args:
        ocr_data: OCR data from Step 2 (doc_id -> ocr_data dict)
        output_dir: Optional directory to save layout JSON results
        limit: Maximum number of documents to process (default: 100)

    Returns:
        Dictionary with doc_id as key, PageLayout as value
    """
    # Randomly select documents to process (if limit is set)
    all_doc_ids = list(ocr_data.keys())
    if limit and limit < len(all_doc_ids):
        doc_ids = random.sample(all_doc_ids, limit)
    else:
        doc_ids = all_doc_ids
    limited_data = {doc_id: ocr_data[doc_id] for doc_id in doc_ids}

    logger.info(f"Starting STEP 3: Layout Understanding for {len(limited_data)} documents (limit: {limit})")

    analyzer = LayoutAnalyzer()
    results = {}

    for doc_id in doc_ids:
        data = ocr_data[doc_id]
        try:
            # Extract OCR blocks
            ocr_blocks_data = data.get('ocr_blocks', [])
            ocr_blocks = []

            for block_data in ocr_blocks_data:
                block = OCRBlock(
                    text=block_data['text'],
                    bbox=tuple(block_data['bbox']),
                    confidence=block_data['confidence']
                )
                ocr_blocks.append(block)

            # Analyze layout
            page_layout = analyzer.analyze_layout(ocr_blocks)
            page_layout.doc_id = data.get('doc_id', doc_id)
            page_layout.page_number = data.get('page_number', 1)

            results[doc_id] = page_layout

            logger.info(f"Layout analysis complete for {doc_id}")

        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            continue

    # Export to JSON if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key, layout in results.items():
            filename = f"{layout.doc_id}_page{layout.page_number}_layout_output.json"
            filepath = os.path.join(output_dir, filename)
            layout.to_json(filepath)

        logger.info(f"Exported {len(results)} layout result files to {output_dir}")

    logger.info(f"STEP 3 complete: Processed {len(results)} pages")

    return results


def demo_layout_analysis(ocr_dir: str = 'ocr_output'):
    """
    Demo function to show layout analysis on OCR data

    Args:
        ocr_dir: Directory containing OCR JSON files
    """
    logger.info("Starting Layout Analysis Demo")

    # Load OCR data
    ocr_data = load_ocr_output(ocr_dir)

    if not ocr_data:
        logger.warning(f"No OCR data found in {ocr_dir}")
        return

    print("\n" + "="*60)
    print("STEP 3: Layout Understanding Demo")
    print("="*60)

    print(f"Loaded OCR data for {len(ocr_data)} documents")

    # Process first document as example
    sample_doc_id = list(ocr_data.keys())[0]
    sample_data = ocr_data[sample_doc_id]

    print(f"\nProcessing sample document: {sample_doc_id}")
    print(f"OCR blocks: {len(sample_data.get('ocr_blocks', []))}")

    # Run layout analysis
    results = process_step3({sample_doc_id: sample_data})

    if sample_doc_id in results:
        layout = results[sample_doc_id]

        print("\nLayout Analysis Results:")
        print(f"  Header blocks: {len(layout.header_blocks)}")
        print(f"  Body blocks: {len(layout.body_blocks)}")
        print(f"  Footer blocks: {len(layout.footer_blocks)}")

        # Show sample content from each region
        if layout.header_blocks:
            print(f"\n  Sample header: '{layout.header_blocks[0].text[:50]}...'")

        if layout.body_blocks:
            print(f"  Sample body: '{layout.body_blocks[0].text[:50]}...'")

        if layout.footer_blocks:
            print(f"  Sample footer: '{layout.footer_blocks[0].text[:50]}...'")

        # Export sample JSON
        json_output = layout.to_json()
        print(f"\n  JSON output length: {len(json_output)} characters")
        print(f"  Sample JSON (first 500 chars): {json_output[:500]}...")

    return results


# Example usage
if __name__ == "__main__":
    import sys

    # Run demo using existing OCR output
    demo_layout_analysis()

    # Test integration with Step 2
    print("\n" + "="*60)
    print("Testing Step 2 + Step 3 Integration")
    print("="*60)

    try:
        # Import Step 2 functions using importlib (module name starts with digit)
        import importlib
        ocr_module = importlib.import_module('2_OCR')
        process_step2 = ocr_module.process_step2

        # Check if OCR output already exists
        ocr_output_dir = "ocr_output"
        if os.path.exists(ocr_output_dir) and any(f.endswith('.json') for f in os.listdir(ocr_output_dir)):
            print(f"\nFound existing OCR output in: {ocr_output_dir}")
            print("Loading existing OCR data instead of re-running Step 1 and Step 2...")

            # Load existing OCR data using Layout's load_ocr_output
            existing_ocr_data = load_ocr_output(ocr_output_dir)
            print(f"Loaded {len(existing_ocr_data)} documents from existing OCR output")

            # Process Step 3 with existing OCR data
            layout_output_dir = "layout_output"
            os.makedirs(layout_output_dir, exist_ok=True)
            step3_results = process_step3(existing_ocr_data, output_dir=layout_output_dir)
            print(f"Step 3: Processed {len(step3_results)} pages")
            print(f"  Results saved to: {layout_output_dir}/")

            # Show sample results
            for key, layout in list(step3_results.items())[:2]:
                print(f"\nDocument: {layout.doc_id}, Page: {layout.page_number}")
                print(f"  Header: {len(layout.header_blocks)}, Body: {len(layout.body_blocks)}, Footer: {len(layout.footer_blocks)}")
        else:
            # OCR output doesn't exist, run full pipeline
            print("\nNo existing OCR output found. Running full pipeline...")

            train_dir = "normalized_output"
            if os.path.exists(train_dir):
                print(f"\nProcessing normalized images from: {train_dir}")

                # Create temp directory for output
                import tempfile
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Process Step 1 (assuming normalized images exist)
                    from .Normalize import process_step1
                    step1_results = process_step1(train_dir, temp_dir)
                    print(f"Step 1: Processed {len(step1_results)} documents")

                    # Process Step 2
                    step2_results = process_step2(step1_results, output_dir=ocr_output_dir)
                    print(f"Step 2: Processed {len(step2_results)} pages")
                    print(f"  Results saved to: {ocr_output_dir}/")

                    # Process Step 3
                    layout_output_dir = "layout_output"
                    os.makedirs(layout_output_dir, exist_ok=True)
                    step3_results = process_step3(step2_results, output_dir=layout_output_dir)
                    print(f"Step 3: Processed {len(step3_results)} pages")
                    print(f"  Results saved to: {layout_output_dir}/")

                    # Show sample results
                    for key, layout in list(step3_results.items())[:2]:
                        print(f"\nDocument: {layout.doc_id}, Page: {layout.page_number}")
                        print(f"  Header: {len(layout.header_blocks)}, Body: {len(layout.body_blocks)}, Footer: {len(layout.footer_blocks)}")
            else:
                print(f"Training directory {train_dir} not found")

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("Run normalize.py and ocr.py first to create Step 1 and Step 2 output")
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()

