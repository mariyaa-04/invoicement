"""
STEP 2: OCR & Text-Layout Extraction (EasyOCR + Tesseract)

Location: utils/20_OCR.py

Purpose of Step 2:
- Convert pixels into weak, layout-aware text signals
- Preserve where text came from (position)
- Handle multiple layouts and languages
- Support both EasyOCR (default) and Tesseract engines
- Stay simple and cheap

Author: Document AI System
Version: 2.1
"""

import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import random
from difflib import SequenceMatcher
import requests
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# VERNACULAR KEYWORDS (Native Script + Transliterated)
# =============================================================================

# Dealer/Company Keywords - English
DEALER_KEYWORDS_ENGLISH = [
    'dealer', 'authorized', 'm/s', 'm/s ', 'tractors', 'motors', 'ltd', 'llp',
    'distributor', 'agency', 'sales', 'service', 'automobiles', 'vehicles',
    'company', 'corporation', 'enterprise', 'showroom', 'outlet', 'branch',
    'partner', 'associate', 'vendor', 'supplier', 'stockist'
]

# Dealer Keywords - Hindi (Native Devanagari)
DEALER_KEYWORDS_HINDI_NATIVE = [
    'डीलर', 'विक्रेता', 'एम/एस', 'मोटर्स', 'लिमिटेड', 'एलएलपी', 'वितरक',
    'एजेंसी', 'बिक्री', 'सर्विस', 'ऑटोमोबाइल', 'वाहन', 'कंपनी', 'निगम',
    'उद्यम', 'शोरूम', 'आउटलेट', 'शाखा', 'भागीदार', 'सहयोगी', 'विक्रेता',
    'आपूर्तिकर्ता', 'स्टॉकिस्ट', 'ट्रैक्टर एजेंसी', 'ट्रैक्टर कंपनी',
    'ट्रैक्टर शोरूम', 'ट्रैक्टर डीलर', 'कृषि यंत्र', 'खेती उपकरण'
]

# Dealer Keywords - Hindi (Transliterated/Romanized - common in OCR)
DEALER_KEYWORDS_HINDI_TRANS = [
    'dealer', 'vikreta', 'vikretA', 'm/s', 'motars', 'limited', 'llp', 'vitarak',
    'agenSee', 'bikree', 'sarvis', 'automoobil', 'vaahan', 'kampani', 'nigham',
    'udyam', 'sharum', 'aotlet', 'shakha', 'bhagidaar', 'sahyogi', 'aapurtikarta',
    'stokist', 'traktar agenSee', 'traktar kampani', 'traktar sharum',
    'traktar dealer', 'krishi yantr', 'kheti upkaran',
    # Common OCR variations
    'deelar', 'dealaar', 'daalar', 'daalaar', 'vkreta', 'vikrta', 'veekreta',
    'motar', 'motor', 'kampany', 'kampanee', 'shropm', 'showrowm', 'shoowroom',
    'agenCee', 'agensi', 'agenCy', 'sarvis', 'servis', 'serivice', 'serves',
    'vikas agent', 'vikas agency', 'traktar', 'tractar', 'trakter', 'traktar'
]

# Dealer Keywords - Gujarati (Native Script)
DEALER_KEYWORDS_GUJARATI_NATIVE = [
    'ડિલર', 'વિક્ર��તા', 'એમ/એસ', 'મોટર્સ', 'લિમિટેડ', 'એલએલપી', 'વિતરક',
    'એજન્સી', 'વિક્રી', 'સર્વિસ', 'ઓટોમોબાઇલ', 'વાહન', 'કંપની', 'નિગમ',
    'ઉદ્યમ', 'શોરૂમ', 'આઉટલेट', 'શાખा', 'भागीदार', 'सहयोगी', 'विक्रेता',
    'аапур्तिकर्ता', 'स्टॉकिस्ट', 'ट्रैक्टर एजेंसी', 'ट्रैक्टर कंपनी',
    'ट्रैक्टर शोरूम', 'ट्रैक्टर डीलर', 'कृषि यंत्र', 'खेती उपकरण',
    'ટ્રાક્ટર', 'એજન્સી', 'ગરમ', 'શોરૂમ', 'કંપની'
]

# Dealer Keywords - Gujarati (Transliterated/Romanized - common in OCR)
DEALER_KEYWORDS_GUJARATI_TRANS = [
    'dealer', 'vikreta', 'vikretA', 'm/s', 'motars', 'limited', 'llp', 'vitarak',
    'agenSee', 'vikree', 'sarvis', 'automobile', 'vahan', 'kampani', 'nigham',
    'udyam', 'sharuum', 'outlet', 'shakha', 'bhagidaar', 'sahyogi', 'vikreta',
    'aapurtikarta', 'stokist', 'traktar', 'trakter', 'traktor',
    # Common OCR variations
    'deelaar', 'daalar', 'vkreta', 'veekreta', 'motar', 'motor',
    'kampany', 'kampanee', 'sharwm', 'showroom', 'agenCee', 'agensi',
    'sarvis', 'servis', 'traktar dealer', 'traktor agency',
    'kisaan', 'krishi', 'agrico', 'tractors', 'motoren', 'motars'
]

# HP/Power Keywords - English
HP_KEYWORDS_ENGLISH = [
    'hp', 'horse', 'power', 'bh[Pp]', 'engine', 'capacity', 'tractor',
    'power output', 'engine power', 'rated power', 'maximum power',
    'bhp', 'metric hp', 'engine capacity', 'displacement'
]

# HP Keywords - Hindi (Native)
HP_KEYWORDS_HINDI_NATIVE = [
    'हॉर्स पावर', 'हॉर्सपावर', 'एचपी', 'भाप', 'पावर', 'शक्ति', 'इंजन',
    'क्षमता', 'ट्रैक्टर', 'मोटर पावर', 'इंजन पावर', 'अधिकतम पावर',
    'रेटेड पावर', 'बीएचपी', 'मीट्रिक एचपी', 'इंजन क्षमता', 'विस्थापन',
    'बल', 'टॉर्क', 'ड्राइव पावर', 'पीटीओ', 'पावर टेक ऑफ'
]

# HP Keywords - Hindi (Transliterated)
HP_KEYWORDS_HINDI_TRANS = [
    'horsepower', 'horsepowar', 'horsepawar', 'hors power', 'hp', 'bhp',
    'power', 'powar', 'pawar', 'shkti', 'shakti', 'enjin', 'engine', 'enjine',
    'capacity', 'kapasiti', 'kmapcity', 'traktar', 'tractar', 'trakter',
    'motor power', 'motar power', 'max power', 'maximum power',
    # Common OCR variations
    'haurs power', 'hors powr', 'hars power', 'horspoar', 'horspwr',
    'AEchPee', 'aichpi', 'aych pi', 'eech pee', 'bhaap', 'bap', 'bhaap',
    'paavar', 'pawer', 'powr', 'shkti', 'shktY', 'enjen', 'enjyn',
    'kamal', 'kammlti', 'toraq', 'torque', 'traktr', 'traktar',
    'PTO', 'pto', 'ptoo', 'power takeoff', 'power take off'
]

# HP Keywords - Gujarati (Native Script)
HP_KEYWORDS_GUJARATI_NATIVE = [
    'હોર્સપાવર', 'HP', 'એચપી', 'ભાપ', 'પાવર', 'શક્તિ', 'એન્જિન',
    'ક્ષમત', 'ટ્રાક્ટર', 'મોટર', 'ટોર્ક', ' PTO', 'પાવર ટેકઓફ',
    'મહત્તમ', 'રेटेड', 'bhp', 'મીટ્રિક'
]

# HP Keywords - Gujarati (Transliterated/Romanized)
HP_KEYWORDS_GUJARATI_TRANS = [
    'horsepower', 'horsepowar', 'horsepawar', 'hors power', 'hp', 'bhp',
    'power', 'powar', 'pawar', 'shkti', 'shakti', 'enjin', 'engine', 'enjine',
    'capacity', 'kapasiti', 'traktar', 'trakter', 'traktor',
    'motor power', 'motar power', 'max power', 'maximum power',
    # Common OCR variations
    'haurs power', 'hors powr', 'horspoar', 'horspwr',
    'AEchPee', 'aichpi', 'aych pi', 'paavar', 'pawer',
    'torque', 'traktr', 'PTO', 'pto', 'power takeoff'
]

# Cost/Price Keywords - English
COST_KEYWORDS_ENGLISH = [
    'total', 'amount', 'price', 'cost', 'rs', '₹', 'rupees', 'grand total',
    'net amount', 'invoice value', 'deal value', 'transaction value',
    'payment', 'billing', 'quotation', 'estimate', 'ex-showroom', 'on-road'
]

# Cost Keywords - Hindi (Native)
COST_KEYWORDS_HINDI_NATIVE = [
    'कुल', 'राशि', 'कीमत', 'मूल्य', 'भुगतान', 'टोटल', 'ग्रैंड टोटल',
    'शुद्ध राशि', 'इनवॉइस मूल्य', 'डील वैल्यू', 'ट्रांजैक्शन वैल्यू',
    'बिल', 'कोटेशन', 'अनुमान', 'एक्स-शोरूम', 'ऑन-रोड', 'दाम',
    'भाव', 'दर', 'वैल्यू', 'खर्चा', 'व्यय'
]

# Cost Keywords - Hindi (Transliterated)
COST_KEYWORDS_HINDI_TRANS = [
    'kul', 'raashi', 'raashee', 'kimat', 'keemat', 'muly', 'moolya',
    'bhugtan', 'bhugataan', 'total', 'grand total', 'net amount',
    'shudh raashi', 'shudh rashi', 'invoice value', 'invois mulya',
    'deal value', 'del valyu', 'transaction value', 'tranzakshan valyu',
    'bill', 'bil', 'quotation', 'kwoteshan', 'estimate', 'eshtimt',
    'ex showroom', 'exsharwm', 'on road', 'onrod', 'on-road',
    # Common OCR variations
    'dam', 'daam', 'dham', 'bhaav', 'bhaav', 'dar', 'daar',
    'kharcha', 'kharchya', 'vyay', 'veay', 'lagat', 'lgaat',
    'rupaiya', 'rupee', 'rupiya', 'rs', 'rpees', 'rs.',
    'mulya', 'moolya', 'qeemat', 'qemat',
    'jma', 'jmaa', 'jmaa'  # Common OCR error for 'raashi'
]

# Cost Keywords - Gujarati (Native Script)
COST_KEYWORDS_GUJARATI_NATIVE = [
    'કુલ', 'રાશિ', 'કિમત', 'મૂલ્ય', 'ભुगतान', 'ટોટલ', 'ગ્રાન્ડ ટોટલ',
    'શુદ્ધ', 'ઇનવoઇસ', 'bill', 'bil', 'દામ', 'ભાવ', 'દર', 'ખર્ચ',
    'भुगतान', 'भाव', 'दर', 'खर्च', 'व्यय', 'राशि', 'कीमत', 'मूल्य',
    '₹', 'rupees', 'rs', 'payment'
]

# Cost Keywords - Gujarati (Transliterated/Romanized)
COST_KEYWORDS_GUJARATI_TRANS = [
    'kul', 'raashi', 'raashee', 'kimat', 'keemat', 'muly', 'moolya',
    'bhugtan', 'bhugataan', 'total', 'grand total', 'net amount',
    'shudh raashi', 'invoice value', 'deal value', 'transaction value',
    'bill', 'bil', 'quotation', 'estimate', 'ex showroom', 'on road',
    # Common OCR variations
    'dam', 'daam', 'dham', 'bhaav', 'bhaav', 'dar', 'daar',
    'kharcha', 'kharchya', 'vyay', 'veay', 'lagat', 'lgaat',
    'rupaiya', 'rupee', 'rupiya', 'rs', 'rpees', 'rs.',
    'mulya', 'moolya', 'qeemat', 'qemat',
    'jma', 'jmaa', 'jmaa'  # Common OCR error for 'raashi'
]

# Combined vernacular keywords for matching
DEALER_KEYWORDS_HINDI = DEALER_KEYWORDS_HINDI_NATIVE + DEALER_KEYWORDS_HINDI_TRANS
HP_KEYWORDS_HINDI = HP_KEYWORDS_HINDI_NATIVE + HP_KEYWORDS_HINDI_TRANS
COST_KEYWORDS_HINDI = COST_KEYWORDS_HINDI_NATIVE + COST_KEYWORDS_HINDI_TRANS

# Gujarati keywords (all categories combined)
DEALER_KEYWORDS_GUJARATI = DEALER_KEYWORDS_GUJARATI_NATIVE + DEALER_KEYWORDS_GUJARATI_TRANS
HP_KEYWORDS_GUJARATI = HP_KEYWORDS_GUJARATI_NATIVE + HP_KEYWORDS_GUJARATI_TRANS
COST_KEYWORDS_GUJARATI = COST_KEYWORDS_GUJARATI_NATIVE + COST_KEYWORDS_GUJARATI_TRANS


@dataclass
class OCRBlock:
    """
    A single OCR text element with its position and confidence.
    
    This is the atomic unit of Step 2 output - no merging, no interpretation.
    
    Enhanced for handwritten field detection:
    - is_handwritten: Indicates if text appears handwritten
    - stroke_width_avg: Average stroke width for handwriting analysis
    - symbol_type: Type of symbol if detected (checkmark, circle, etc.)
    """
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float  # 0.0 to 1.0
    is_handwritten: bool = False  # Whether text appears handwritten
    stroke_width_avg: float = 0.0  # Average stroke width for analysis
    symbol_type: str = None  # Type of symbol if detected (checkmark, tick, cross, etc.)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'bbox': list(self.bbox),
            'confidence': self.confidence,
            'is_handwritten': self.is_handwritten,
            'stroke_width_avg': self.stroke_width_avg,
            'symbol_type': self.symbol_type
        }


@dataclass
class PageOCRData:
    """
    OCR data for a single page.
    
    This is the core output of Step 2 - layout-agnostic text extraction.
    """
    doc_id: str
    page_number: int
    ocr_blocks: List[OCRBlock] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_id': self.doc_id,
            'page_number': self.page_number,
            'ocr_blocks': [block.to_dict() for block in self.ocr_blocks]
        }
    
    def to_json(self, filepath: str = None) -> str:
        """Export to JSON string or file"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"OCR data saved to {filepath}")
        return json_str


class OCREngine:
    """
    Pure OCR Engine using Tesseract.
    
    Responsibilities:
    - Read text from images
    - Preserve position (bounding boxes)
    - Handle multiple languages
    - Return confidence scores
    
    Non-responsibilities:
    - Understanding the content
    - Classifying fields
    - Cleaning text
    
    Enhanced for handwritten field detection:
    - Detects handwriting characteristics
    - Identifies checkmarks and symbols
    - Provides stroke width analysis
    """
    
    # Language codes for Tesseract
    LANGUAGE_CODES = {
        'eng': 'eng',
        'english': 'eng',
        'hin': 'hin',
        'hindi': 'hin',
        'guj': 'guj',
        'gujarati': 'guj'
    }
    
    # Default Tesseract configuration
    DEFAULT_CONFIG = '--psm 6'  # Treat image as a single uniform block of text
    
    def __init__(self, languages: List[str] = ['eng'], config: str = None):
        """
        Initialize OCR Engine
        
        Args:
            languages: List of languages to use ['eng', 'hin', 'guj']
            config: Tesseract config string (default: '--psm 6')
        """
        self.languages = [self.LANGUAGE_CODES.get(lang.lower(), 'eng') for lang in languages]
        self.language_code = '+'.join(set(self.languages))  # Remove duplicates, join with +
        self.config = config or self.DEFAULT_CONFIG
        
        # Verify Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.warning(f"Tesseract not found or error: {e}")
            logger.info("Please install Tesseract: https://github.com/tesseract-ocr/tesseract")
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract full text from image using Tesseract OCR.
        
        Args:
            image: Input image (numpy array, BGR or grayscale)
            
        Returns:
            Extracted text as string
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.language_code,
                config=self.config
            )
            return text.strip()
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return ""
    
    def _analyze_stroke_width(self, region: np.ndarray) -> Dict:
        """
        Analyze stroke width patterns to detect handwriting.
        
        Args:
            region: Image region to analyze
            
        Returns:
            Dictionary with 'is_handwritten' bool, confidence, and avg_width
        """
        if region.size == 0:
            return {'is_handwritten': False, 'confidence': 0.5, 'avg_width': 0}
        
        # Apply edge detection
        edges = cv2.Canny(region, 50, 150)
        
        # Connected components analysis
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.threshold(region, 127, 255, cv2.THRESH_BINARY)[1]
        )
        
        if labels is None or len(stats) <= 1:
            return {'is_handwritten': False, 'confidence': 0.5, 'avg_width': 0}
        
        # Analyze component sizes
        component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        if len(component_sizes) == 0:
            return {'is_handwritten': False, 'confidence': 0.5, 'avg_width': 0}
        
        size_variance = np.var(component_sizes)
        size_mean = np.mean(component_sizes)
        
        # Normalize variance by mean
        normalized_variance = size_variance / (size_mean ** 2) if size_mean > 0 else 0
        
        # Handwriting tends to have higher normalized variance
        is_handwritten = normalized_variance > 0.1
        
        # Calculate confidence
        confidence = min(0.5 + normalized_variance * 2, 0.9)
        
        # Estimate average stroke width
        avg_width = size_mean ** 0.5 if size_mean > 0 else 0
        
        return {
            'is_handwritten': is_handwritten,
            'confidence': confidence,
            'avg_width': avg_width
        }
    
    def _detect_symbol_type(self, text: str, bbox: Tuple[int, int, int, int],
                            image_shape: Tuple[int, int]) -> str:
        """
        Detect if text is a symbol type (checkmark, tick, cross, etc.)
        
        Args:
            text: Extracted text
            bbox: Bounding box
            image_shape: Shape of the original image
            
        Returns:
            Symbol type string or None
        """
        # Check for checkmark symbols
        checkmark_patterns = ['✓', '✔', '☑', '√', '✗', '✘', 'X', 'x', '☑', '☐']
        
        if text.strip() in checkmark_patterns:
            if text in ['✓', '✔', '√']:
                return 'tick'
            elif text in ['✗', '✘', 'X', 'x']:
                return 'cross'
            elif text in ['☑']:
                return 'circle_filled'
            elif text in ['☐']:
                return 'circle_empty'
        
        # Check for single characters that might be handwritten marks
        if len(text.strip()) == 1 and text.strip().isalpha():
            # Single letter might be a handwritten mark
            return 'handwritten_mark'
        
        return None
    
    def extract_ocr_blocks(self, image: np.ndarray) -> List[OCRBlock]:
        """
        Extract text with bounding boxes and confidence scores.
        
        This is the CORE method of Step 2. It returns raw OCR blocks
        without any interpretation or field extraction.
        
        Enhanced for handwritten field detection:
        - Detects handwriting characteristics
        - Identifies checkmarks and symbols
        - Provides stroke width analysis
        
        Args:
            image: Input image (numpy array, BGR or grayscale)
            
        Returns:
            List of OCRBlock objects, each containing:
            - text: The detected text
            - bbox: Bounding box (x1, y1, x2, y2)
            - confidence: Confidence score (0.0 to 1.0)
            - is_handwritten: Whether text appears handwritten
            - stroke_width_avg: Average stroke width for analysis
            - symbol_type: Type of symbol if detected
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        image_shape = (h, w)
        
        try:
            # Get detailed OCR data from Tesseract
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language_code,
                config=self.config,
                output_type=Output.DICT
            )
            
            # Convert to OCRBlocks
            ocr_blocks = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if text:  # Only include non-empty text
                    bbox = (
                        ocr_data['left'][i],
                        ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i]
                    )
                    conf = ocr_data['conf'][i]
                    
                    # Handle confidence -1 (not found)
                    if conf == -1:
                        conf = 0.0
                    else:
                        conf = float(conf) / 100.0  # Normalize to 0-1
                    
                    # Analyze stroke width for handwriting detection
                    x1, y1, x2, y2 = bbox
                    # Ensure bbox is within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        region = gray[y1:y2, x1:x2]
                        stroke_analysis = self._analyze_stroke_width(region)
                    else:
                        stroke_analysis = {'is_handwritten': False, 'confidence': 0.5, 'avg_width': 0}
                    
                    # Detect symbol type
                    symbol_type = self._detect_symbol_type(text, bbox, image_shape)
                    
                    ocr_blocks.append(OCRBlock(
                        text=text,
                        bbox=bbox,
                        confidence=conf,
                        is_handwritten=stroke_analysis['is_handwritten'],
                        stroke_width_avg=stroke_analysis['avg_width'],
                        symbol_type=symbol_type
                    ))
            
            return ocr_blocks
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return []


class EasyOCREngine:
    """
    OCR Engine using EasyOCR (PyTorch-based deep learning OCR).
    
    Responsibilities:
    - Read text from images using deep learning
    - Preserve position (bounding boxes)
    - Handle multiple languages
    - Return confidence scores
    
    Advantages over Tesseract:
    - Better at handling complex fonts
    - Better at handwritten text
    - No Tesseract installation required
    
    Non-responsibilities:
    - Understanding the content
    - Classifying fields
    - Cleaning text
    """
    
    # Language codes for EasyOCR
    LANGUAGE_CODES = {
        'eng': 'en',
        'english': 'en',
        'hin': 'hi',
        'hindi': 'hi',
        'guj': 'gu',
        'gujarati': 'gu'
    }
    
    def __init__(self, languages: List[str] = ['eng'], use_gpu: bool = False):
        """
        Initialize EasyOCR Engine
        
        Args:
            languages: List of languages to use ['en', 'hi', 'gu']
            use_gpu: Whether to use GPU acceleration
        """
        self.languages = [self.LANGUAGE_CODES.get(lang.lower(), 'en') for lang in languages]
        self.use_gpu = use_gpu
        self.reader = None
        
        # Try to initialize EasyOCR
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except Exception as e:
            logger.warning(f"EasyOCR initialization error: {e}")
            logger.info("Install EasyOCR: pip install easyocr")
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract full text from image using EasyOCR.
        
        Args:
            image: Input image (numpy array, BGR or RGB)
            
        Returns:
            Extracted text as string (joined by newlines)
        """
        if self.reader is None:
            logger.error("EasyOCR not initialized")
            return ""
        
        # EasyOCR expects RGB image
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        try:
            results = self.reader.readtext(rgb_image)
            text_lines = [result[1] for result in results]
            return '\n'.join(text_lines).strip()
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            return ""
    
    def extract_ocr_blocks(self, image: np.ndarray) -> List[OCRBlock]:
        """
        Extract text with bounding boxes and confidence scores using EasyOCR.
        
        Args:
            image: Input image (numpy array, BGR or RGB)
            
        Returns:
            List of OCRBlock objects
        """
        if self.reader is None:
            logger.error("EasyOCR not initialized")
            return []
        
        # EasyOCR expects RGB image
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        try:
            results = self.reader.readtext(rgb_image)
            
            ocr_blocks = []
            for result in results:
                # EasyOCR result format: [bbox, text, confidence]
                bbox = result[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text = result[1].strip()
                confidence = result[2]
                
                if text:  # Only include non-empty text
                    # Convert bbox to (x1, y1, x2, y2) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    ocr_blocks.append(OCRBlock(
                        text=text,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(confidence)
                    ))
            
            return ocr_blocks
            
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            return []


class HybridOCREngine:
    """
    Hybrid OCR Engine that combines EasyOCR and Tesseract results.
    
    This class provides:
    - Option to use EasyOCR (default) or Tesseract
    - Combined results from both engines
    - Automatic fallback if one engine fails
    
    Usage:
        engine = HybridOCREngine(engine='easyocr')    # EasyOCR only (default)
        engine = HybridOCREngine(engine='tesseract')  # Tesseract only
        engine = HybridOCREngine(engine='both')       # Combine both
    """
    
    def __init__(self, engine: str = 'tesseract', languages: List[str] = ['eng'],
                 use_gpu: bool = False):
        """
        Initialize Hybrid OCR Engine
        
        Args:
            engine: 'tesseract', 'easyocr', or 'both'
            languages: List of languages
            use_gpu: Use GPU for EasyOCR (if applicable)
        """
        self.engine_type = engine
        self.tesseract_engine = None
        self.easyocr_engine = None
        
        if engine in ['tesseract', 'both']:
            self.tesseract_engine = OCREngine(languages)
        
        if engine in ['easyocr', 'both']:
            self.easyocr_engine = EasyOCREngine(languages, use_gpu)
    
    def extract_ocr_blocks(self, image: np.ndarray,
                           merge_strategy: str = 'union') -> List[OCRBlock]:
        """
        Extract OCR blocks using the configured engine(s).
        
        Args:
            image: Input image
            merge_strategy: 'union', 'intersection', or 'tesseract_preferred'
                           (only used when engine='both')
        
        Returns:
            List of OCRBlock objects
        """
        if self.tesseract_engine and self.easyocr_engine:
            # Both engines available - combine results
            tesseract_blocks = self.tesseract_engine.extract_ocr_blocks(image)
            easyocr_blocks = self.easyocr_engine.extract_ocr_blocks(image)
            
            return self._merge_blocks(tesseract_blocks, easyocr_blocks,
                                     strategy=merge_strategy)
        elif self.tesseract_engine:
            return self.tesseract_engine.extract_ocr_blocks(image)
        elif self.easyocr_engine:
            return self.easyocr_engine.extract_ocr_blocks(image)
        else:
            logger.error("No OCR engine available")
            return []
    
    def _merge_blocks(self, blocks1: List[OCRBlock], blocks2: List[OCRBlock],
                      strategy: str = 'union') -> List[OCRBlock]:
        """
        Merge OCR blocks from two engines.
        
        Args:
            blocks1: Blocks from first engine
            blocks2: Blocks from second engine
            strategy: 'union' (all unique), 'intersection' (common), 'tesseract_preferred'
        
        Returns:
            Merged list of OCR blocks
        """
        if strategy == 'tesseract_preferred':
            # Use Tesseract as primary, fill gaps with EasyOCR
            result = list(blocks1)
            tesseract_texts = {(b.text.lower(), b.bbox) for b in blocks1}
            
            for block in blocks2:
                key = (block.text.lower(), block.bbox)
                if key not in tesseract_texts:
                    result.append(block)
            
            return result
        
        elif strategy == 'intersection':
            # Only keep blocks that appear in both
            result = []
            tesseract_dict = {(b.text.lower(), b.bbox): b for b in blocks1}
            
            for block in blocks2:
                key = (block.text.lower(), block.bbox)
                if key in tesseract_dict:
                    # Average confidences
                    tesseract_block = tesseract_dict[key]
                    merged_conf = (tesseract_block.confidence + block.confidence) / 2
                    result.append(OCRBlock(
                        text=block.text,
                        bbox=block.bbox,
                        confidence=merged_conf
                    ))
            
            return result
        
        else:  # 'union'
            # All unique blocks from both engines
            result = list(blocks1)
            tesseract_dict = {(b.text.lower(), b.bbox): b for b in blocks1}
            
            for block in blocks2:
                key = (block.text.lower(), block.bbox)
                if key not in tesseract_dict:
                    result.append(block)
            
            return result


class TrOCREngine:
    """
    OCR Engine using TrOCR (Transformer OCR) from Microsoft/Hugging Face.
    
    TrOCR uses a vision transformer encoder-decoder architecture for OCR.
    It's particularly good at printed text with structured layouts.
    
    Responsibilities:
    - Read text from images using Transformer-based OCR
    - Handle line-by-line text recognition
    - Return confidence scores
    
    Advantages:
    - State-of-the-art performance on printed text
    - No external dependencies like Tesseract
    - Learns spatial relationships between text elements
    
    Non-responsibilities:
    - Handling multiple languages (best for English)
    - Understanding the content
    - Classifying fields
    """

    # Model variants available
    MODEL_SIZES = {
        'small': 'microsoft/trocr-small-printed',
        'base': 'microsoft/trocr-base-printed',
        'large': 'microsoft/trocr-large-printed'
    }
    
    def __init__(self, model_size: str = 'base', use_gpu: bool = False, 
                 local_files_only: bool = False):
        """
        Initialize TrOCR Engine
        
        Args:
            model_size: 'small', 'base', or 'large'
            use_gpu: Whether to use GPU acceleration
            local_files_only: Use cached models only (no download)
        """
        self.model_size = model_size
        self.use_gpu = use_gpu
        self.processor = None
        self.model = None
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load model
        self._load_model(local_files_only)
    
    def _load_model(self, local_files_only: bool = False):
        """Load the TrOCR processor and model"""
        try:
            model_name = self.MODEL_SIZES.get(self.model_size, 'microsoft/trocr-base-printed')
            
            logger.info(f"Loading TrOCR model: {model_name} on {self.device}")
            
            # Download the model if not already cached
            self.processor = TrOCRProcessor.from_pretrained(
                model_name,
                local_files_only=local_files_only
            )
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                local_files_only=local_files_only
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"TrOCR model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading TrOCR model: {e}")
            logger.info("Make sure you have internet connection for first run, ")
            logger.info("or use local_files_only=True if model is cached")
            raise
    
    def _download_model_manually(self, model_name: str, output_dir: str = "./trocr_model"):
        """
        Manually download TrOCR model files from Hugging Face.
        
        This is useful when you want to download the model files
        explicitly instead of letting transformers handle it.
        
        Args:
            model_name: Hugging Face model name
            output_dir: Local directory to save model
        """
        import os
        from huggingface_hub import snapshot_download
        
        logger.info(f"Downloading TrOCR model: {model_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Download model files
        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model downloaded to: {output_dir}")
        return output_dir
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract full text from image using TrOCR.
        
        TrOCR works best on line-by-line recognition, so we may need
        to segment the image into text lines first for best results.
        
        Args:
            image: Input image (numpy array, BGR or RGB)
            
        Returns:
            Extracted text as string
        """
        if self.processor is None or self.model is None:
            logger.error("TrOCR model not loaded")
            return ""
        
        # Convert to RGB if needed (TrOCR expects RGB)
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        try:
            # Process image
            pixel_values = self.processor(
                images=pil_image, 
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode to text
            text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"TrOCR extraction error: {e}")
            return ""
    
    def extract_text_with_lines(self, image: np.ndarray, 
                                 line_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Extract text from pre-segmented line images with confidence scores.
        
        This is the recommended approach for best results with TrOCR.
        Segment the image into individual text lines first, then process each.
        
        Args:
            image: Original full image (unused, for reference)
            line_images: List of cropped line images (numpy arrays)
            
        Returns:
            List of tuples: (text, confidence_score)
        """
        if self.processor is None or self.model is None:
            logger.error("TrOCR model not loaded")
            return []
        
        results = []
        
        for line_image in line_images:
            if len(line_image.shape) == 3:
                rgb_line = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_line = line_image
            
            pil_line = Image.fromarray(rgb_line)
            
            try:
                # Process line
                pixel_values = self.processor(
                    images=pil_line,
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Generate with scores
                with torch.no_grad():
                    generated = self.model.generate(
                        pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True
                    )
                
                # Get generated text
                generated_ids = generated.sequences
                text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                # Calculate confidence from softmax of first token
                probs = torch.nn.functional.softmax(generated.scores[0], dim=-1)
                max_prob = torch.max(probs).item()
                
                if text:
                    results.append((text, max_prob))
                    
            except Exception as e:
                logger.error(f"Error processing line: {e}")
                continue
        
        return results
    
    def extract_ocr_blocks(self, image: np.ndarray) -> List[OCRBlock]:
        """
        Extract text with bounding boxes and confidence scores.
        
        Note: TrOCR doesn't provide bounding boxes directly.
        This implementation uses a simple line segmentation approach.
        For better results, combine with another engine (like Tesseract)
        for bounding box detection.
        
        Args:
            image: Input image (numpy array, BGR or RGB)
            
        Returns:
            List of OCRBlock objects
        """
        if self.processor is None or self.model is None:
            logger.error("TrOCR model not loaded")
            return []
        
        # Simple approach: Use whole image for text extraction
        # Note: TrOCR doesn't detect bounding boxes, so we return
        # a single block with the full image as bbox
        
        text = self.extract_text(image)
        
        if text:
            # Return single block covering full image
            h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
            return [OCRBlock(
                text=text,
                bbox=(0, 0, w, h),
                confidence=0.85  # Default confidence for TrOCR
            )]
        
        return []
    
    def extract_ocr_blocks_from_lines(self, line_images: List[np.ndarray],
                                       bboxes: List[Tuple[int, int, int, int]]) -> List[OCRBlock]:
        """
        Extract OCR blocks from pre-segmented lines with known bounding boxes.
        
        This is the recommended method when you have line segmentation
        from another source (like Tesseract or EasyOCR).
        
        Args:
            line_images: List of cropped line images
            bboxes: List of bounding boxes corresponding to each line
            
        Returns:
            List of OCRBlock objects
        """
        if self.processor is None or self.model is None:
            logger.error("TrOCR model not loaded")
            return []
        
        if len(line_images) != len(bboxes):
            logger.error(f"Mismatch: {len(line_images)} images vs {len(bboxes)} bboxes")
            return []
        
        ocr_blocks = []
        
        for line_image, bbox in zip(line_images, bboxes):
            results = self.extract_text_with_lines(image, [line_image])
            
            for text, confidence in results:
                ocr_blocks.append(OCRBlock(
                    text=text,
                    bbox=bbox,
                    confidence=confidence
                ))
        
        return ocr_blocks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_name': self.MODEL_SIZES.get(self.model_size, 'unknown'),
            'model_size': self.model_size,
            'device': self.device,
            'gpu_available': torch.cuda.is_available() if self.use_gpu else False,
            'processor_loaded': self.processor is not None,
            'model_loaded': self.model is not None
        }


def download_troc_model(model_size: str = 'base', output_dir: str = None) -> str:
    """
    Download TrOCR model files to a local directory.
    
    This function downloads the model files explicitly so they can be
    used offline or in air-gapped environments.
    
    Args:
        model_size: 'small', 'base', or 'large'
        output_dir: Directory to save model (default: ./trocr_{size})
        
    Returns:
        Path to downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
        
        model_name = TrOCREngine.MODEL_SIZES.get(model_size, 'microsoft/trocr-base-printed')
        
        if output_dir is None:
            output_dir = f"./trocr_{model_size}"
        
        logger.info(f"Downloading TrOCR model to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download model files using huggingface_hub
        snapshot_download(
            repo_id=model_name,
            repo_type="model",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model saved to {output_dir}")
        
        # Also save model size info
        info_file = os.path.join(output_dir, "model_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Size: {model_size}\n")
        
        return output_dir
        
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.info("Install: pip install transformers huggingface_hub")
        raise
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise


class InvoiceOCRProcessor:
    """
    Main processor for invoice OCR extraction.

    This is Step 2 of the document pipeline.
    Takes normalized images from Step 1 and produces layout-aware OCR blocks.

    Input (from Step 1):
    - doc_id: Document identifier
    - page_number: Page number
    - normalized_image: Preprocessed image

    Output:
    - PageOCRData with raw OCR blocks (text + bbox + confidence)
    """

    def __init__(self, engine: str = 'easyocr', languages: List[str] = ['eng', 'hin'],
                 config: str = None, use_gpu: bool = False):
        """
        Initialize the processor

        Args:
            engine: OCR engine type - 'trocr+easyocr' (default), 'easyocr', 'tesseract', or 'both'
            languages: Languages for OCR ['eng', 'hin', 'guj']
            config: Tesseract config (not used for easyocr/trocr)
            use_gpu: Use GPU for EasyOCR and TrOCR (if applicable)
        """
        self.engines = {}
        self.engine_type = engine

        # Initialize engines based on type
        if engine == 'trocr+easyocr':
            # TrOCR for English, EasyOCR for regional languages
            if 'eng' in [lang.lower() for lang in languages]:
                self.engines['trocr'] = TrOCREngine(model_size="base", use_gpu=use_gpu)

            regional_langs = [lang for lang in languages if lang.lower() != 'eng']
            if regional_langs:
                self.engines['easyocr'] = EasyOCREngine(regional_langs, use_gpu)

        elif engine == 'easyocr':
            self.engines['easyocr'] = EasyOCREngine(languages, use_gpu)

        elif engine == 'tesseract':
            self.engines['tesseract'] = OCREngine(languages, config)

        elif engine == 'both':
            self.engines['tesseract'] = OCREngine(languages, config)
            self.engines['easyocr'] = EasyOCREngine(languages, use_gpu)

        else:
            # Default to trocr+easyocr
            self.engines['trocr'] = TrOCREngine(model_size="base", use_gpu=use_gpu)
            regional_langs = [lang for lang in languages if lang.lower() != 'eng']
            if regional_langs:
                self.engines['easyocr'] = EasyOCREngine(regional_langs, use_gpu)

        logger.info(f"Initialized OCR engines: {list(self.engines.keys())}")
    
    def process_page(self, doc_id: str, page_number: int,
                     image: np.ndarray) -> PageOCRData:
        """
        Process a single page from Step 1 output using multiple engines.

        Args:
            doc_id: Document identifier
            page_number: Page number
            image: Normalized image from Step 1

        Returns:
            PageOCRData object with raw OCR blocks
        """
        if image is None:
            logger.warning(f"No image provided for document {doc_id}, page {page_number}")
            return PageOCRData(
                doc_id=doc_id,
                page_number=page_number,
                ocr_blocks=[]
            )

        logger.info(f"Processing OCR ({self.engine_type}) for document {doc_id}, page {page_number}")

        all_blocks = []

        # Process with each engine
        for engine_name, engine in self.engines.items():
            try:
                logger.info(f"Running {engine_name} OCR...")
                blocks = engine.extract_ocr_blocks(image)

                # Add engine source to each block for debugging
                for block in blocks:
                    block.text = f"[{engine_name}] {block.text}"

                all_blocks.extend(blocks)
                logger.info(f"{engine_name}: extracted {len(blocks)} blocks")

            except Exception as e:
                logger.error(f"Error with {engine_name}: {e}")
                continue

        # Remove duplicates and sort by position
        unique_blocks = self._deduplicate_blocks(all_blocks)

        logger.info(f"OCR complete for {doc_id}_page{page_number}: "
                   f"{len(unique_blocks)} unique text blocks extracted")

        return PageOCRData(
            doc_id=doc_id,
            page_number=page_number,
            ocr_blocks=unique_blocks
        )

    def _deduplicate_blocks(self, blocks: List[OCRBlock]) -> List[OCRBlock]:
        """
        Remove duplicate OCR blocks based on similar text and position.

        Args:
            blocks: List of OCR blocks from multiple engines

        Returns:
            Deduplicated list of blocks
        """
        if not blocks:
            return blocks

        # Sort by confidence (highest first)
        sorted_blocks = sorted(blocks, key=lambda x: x.confidence, reverse=True)
        unique_blocks = []

        for block in sorted_blocks:
            # Check if similar block already exists
            is_duplicate = False
            for existing in unique_blocks:
                # Check text similarity and bbox overlap
                text_similar = self._text_similarity(block.text, existing.text)
                bbox_overlap = self._bbox_overlap(block.bbox, existing.bbox)

                if text_similar > 0.8 and bbox_overlap > 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_blocks.append(block)

        return unique_blocks

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity ratio."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _bbox_overlap(self, bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate bounding box overlap ratio (IoU).

        Args:
            bbox1, bbox2: Bounding boxes as (x1, y1, x2, y2)

        Returns:
            IoU ratio (0.0 to 1.0)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No overlap

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


def process_step2(step1_output: Dict[str, List[Dict]],
                  languages: List[str] = ['eng', 'hin'],
                  engine: str = 'easyocr',
                  output_dir: str = None,
                  use_gpu: bool = False,
                  limit: int = 100) -> Dict[str, PageOCRData]:
    """
    STEP 2: OCR & Text-Layout Extraction - Main processing function
    
    This is the main entry point for Step 2 processing.
    Takes Step 1 output and returns layout-aware OCR blocks.
    
    INPUT (from Step 1):
    {
        "172610467": [
            {
                "invoice_id": "172610467",
                "page_number": 1,
                "normalized_image": <numpy array>,
                "original_path": "path/to/image.jpg"
            }
        ],
        ...
    }
    
    OUTPUT:
    {
        "172610467_page1": <PageOCRData object>,
        ...
    }
    
    Where PageOCRData has:
    {
        "doc_id": "172610467",
        "page_number": 1,
        "ocr_blocks": [
            {"text": "IDFC FIRST BANK", "bbox": [x1, y1, x2, y2], "confidence": 0.82},
            {"text": "48 HP", "bbox": [...], "confidence": 0.91},
            ...
        ]
    }
    
    Args:
        step1_output: Output from process_step1 (normalized invoice data)
        languages: Languages for OCR ['eng', 'hin', 'guj']
        engine: OCR engine - 'tesseract', 'easyocr', or 'both'
        output_dir: Optional directory to save JSON results
        use_gpu: Use GPU for EasyOCR (if applicable)
        limit: Maximum number of images to process (default: 100)
        
    Returns:
        Dictionary with "doc_id_pageN" as key, PageOCRData as value
    """
    # Randomly select documents to process (if limit is set)
    all_doc_ids = list(step1_output.keys())
    if limit and limit < len(all_doc_ids):
        doc_ids = random.sample(all_doc_ids, limit)
    else:
        doc_ids = all_doc_ids
    limited_output = {doc_id: step1_output[doc_id] for doc_id in doc_ids}
    
    logger.info(f"Starting STEP 2: OCR Strategy ({engine}) for {len(limited_output)} documents (limit: {limit})")
    
    # Initialize processor with specified engine
    processor = InvoiceOCRProcessor(engine=engine, languages=languages, use_gpu=use_gpu)
    
    # Store results
    results = {}
    
    # Process all documents and pages
    for doc_id in doc_ids:
        pages = step1_output[doc_id]
        logger.info(f"Processing document {doc_id} with {len(pages)} pages")
        
        for page_data in pages:
            doc_id_val = page_data.get('invoice_id', doc_id)
            page_num = page_data.get('page_number', 1)
            image = page_data.get('normalized_image')
            
            # Process the page
            ocr_result = processor.process_page(doc_id_val, page_num, image)
            
            # Store with unique key
            key = f"{doc_id_val}_page{page_num}"
            results[key] = ocr_result
    
    # Export to JSON if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for key, ocr_data in results.items():
            filename = f"{ocr_data.doc_id}_page{ocr_data.page_number}_ocr_output.json"
            filepath = os.path.join(output_dir, filename)
            ocr_data.to_json(filepath)
        
        logger.info(f"Exported {len(results)} OCR result files to {output_dir}")
    
    logger.info(f"STEP 2 complete: Processed {len(results)} pages")
    
    return results


def demo_ocr_extraction(image_path: str = None, languages: List[str] = ['eng']):
    """
    Demo function to show OCR extraction on a sample image
    
    Args:
        image_path: Path to sample invoice image
        languages: Languages for OCR
    """
    # Create sample image if no path provided
    if image_path is None:
        # Create a sample invoice-like image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add text (simulating invoice content)
        cv2.putText(img, "INVOICE", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(img, "Dealer: ABC Motors Ltd", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Model: HP-200", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Horse Power: 50", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Cost: Rs. 50,000", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(img, "Date: 15/01/2025", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        image_path = "/tmp/sample_invoice.png"
        cv2.imwrite(image_path, img)
        logger.info(f"Created sample image: {image_path}")
    else:
        # Load existing image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return
    
    # Initialize OCR engine (EasyOCR - default)
    ocr_engine = EasyOCREngine(languages=languages)
    
    # Extract full text
    text = ocr_engine.extract_text(img)
    
    # Extract OCR blocks (THE STEP 2 OUTPUT)
    ocr_blocks = ocr_engine.extract_ocr_blocks(img)
    
    print("\n" + "="*60)
    print("STEP 2: OCR & Text-Layout Extraction Demo")
    print("="*60)
    
    print(f"\nLanguages: {languages}")
    print(f"Image: {image_path}")
    
    print(f"\n--- Raw Text ---")
    print(text if text else "(no text detected)")
    
    print(f"\n--- OCR Blocks (Layout-Aware) ---")
    print(f"Total blocks: {len(ocr_blocks)}")
    for i, block in enumerate(ocr_blocks[:15]):  # Show first 15
        print(f"  [{i}] '{block.text}' at {block.bbox}, conf: {block.confidence:.2f}")
    if len(ocr_blocks) > 15:
        print(f"  ... and {len(ocr_blocks) - 15} more blocks")
    
    # Create PageOCRData (the Step 2 output format)
    page_ocr = PageOCRData(
        doc_id="demo_invoice",
        page_number=1,
        ocr_blocks=ocr_blocks
    )
    
    # Export to JSON
    json_output = page_ocr.to_json()
    print(f"\n--- JSON Output (Step 2 Format) ---")
    print(json_output[:800] + "..." if len(json_output) > 800 else json_output)
    
    return page_ocr


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for sample invoice images
    normalized_dir = "normalized_output"
    
    if os.path.exists(normalized_dir) and os.listdir(normalized_dir):
        print("Found normalized images. Processing sample...")
        
        # Get first normalized image
        sample_image = os.path.join(normalized_dir, os.listdir(normalized_dir)[0])
        demo_ocr_extraction(sample_image)
    else:
        print("No normalized images found. Running demo with sample image...")
        demo_ocr_extraction()
    
    # Also test integration with Step 1
    print("\n" + "="*60)
    print("Testing Step 1 + Step 2 Integration")
    print("="*60)
    
    try:
        # Import Step 1 functions
        from .Normalize import process_step1
        
        # Try to process a small subset
        train_dir = "normalized_output"
        if os.path.exists(train_dir):
            print(f"\nProcessing training directory: {train_dir}")
            
            # Create temp directory for output
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process Step 1
                step1_results = process_step1(train_dir, temp_dir)
                print(f"Step 1: Processed {len(step1_results)} documents")
                
                # Create output directory for OCR results
                ocr_output_dir = "ocr_output"
                os.makedirs(ocr_output_dir, exist_ok=True)
                
                # Process Step 2 with output directory
                step2_results = process_step2(step1_results, languages=['eng'], output_dir=ocr_output_dir)
                print(f"Step 2: Processed {len(step2_results)} pages")
                print(f"  Results saved to: {ocr_output_dir}/")
                
                # Show sample results
                for key, ocr_data in list(step2_results.items())[:2]:
                    print(f"\nDocument: {ocr_data.doc_id}, Page: {ocr_data.page_number}")
                    print(f"  OCR Blocks: {len(ocr_data.ocr_blocks)}")
                    
                    # Show first few blocks as sample
                    if ocr_data.ocr_blocks:
                        print(f"  Sample blocks:")
                        for block in ocr_data.ocr_blocks[:5]:
                            print(f"    - '{block.text}' at {block.bbox}")
        else:
            print(f"Training directory {train_dir} not found")
            
    except ImportError as e:
        print(f"Could not import Normalize module: {e}")
        print("Run normalize.py first to create Step 1 output")
    except Exception as e:
        print(f"Error during integration test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Check for invoice images in 'invoices' folder (flat structure)
    invoices_dir = "invoices"
    
    if os.path.exists(invoices_dir) and os.listdir(invoices_dir):
        print(f"Found {len(os.listdir(invoices_dir))} invoice images in '{invoices_dir}' folder")
        
        # Import Step 1 functions for preprocessing
        from .Normalize import process_step1
        
        print("\n" + "="*60)
        print("Testing Step 1 + Step 2 Integration with 'invoices' folder")
        print("="*60)
        
        try:
            # Process Step 1 on invoices folder
            print(f"\nProcessing invoices directory: {invoices_dir}")
            step1_results = process_step1(invoices_dir)
            print(f"Step 1: Processed {len(step1_results)} invoices")
            
            # Show sample result
            sample_invoice = list(step1_results.keys())[0]
            print(f"Sample invoice: {sample_invoice}")
            print(f"  Pages: {len(step1_results[sample_invoice])}")
            
            # Create output directory for OCR results
            ocr_output_dir = "ocr_output"
            os.makedirs(ocr_output_dir, exist_ok=True)
            
            # Process Step 2 with output directory
            step2_results = process_step2(step1_results, languages=['eng'], output_dir=ocr_output_dir)
            print(f"Step 2: Processed {len(step2_results)} pages")
            print(f"  Results saved to: {ocr_output_dir}/")
            
            # Show sample results
            for key, ocr_data in list(step2_results.items())[:2]:
                print(f"\nDocument: {ocr_data.doc_id}, Page: {ocr_data.page_number}")
                print(f"  OCR Blocks: {len(ocr_data.ocr_blocks)}")
                
                # Show first few blocks as sample
                if ocr_data.ocr_blocks:
                    print(f"  Sample blocks:")
                    for block in ocr_data.ocr_blocks[:5]:
                        print(f"    - '{block.text}' at {block.bbox}")
                        
        except Exception as e:
            print(f"Error during integration test: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No invoice images found in 'invoices' folder. Running demo with sample image...")
        demo_ocr_extraction()

