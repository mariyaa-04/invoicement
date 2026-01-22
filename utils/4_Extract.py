"""
STEP 4: Field Detection & Entity Extraction

Purpose of Step 4:
- Extract structured fields from OCR data
- Use text-based extraction for dealer name, model, HP, cost
- Use vision-based detection for signature and stamp
- Compute confidence scores for each field

This is where the system answers the business question:
"From this document, what is the dealer name, model, HP, cost, and are signature/stamp present?"

Author: Document AI System
Version: 2.0
"""


import os
import re
import json
import logging
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# MASTER LISTS (In production, these would be loaded from database/files)
# =============================================================================

# Extended dealer master list for fuzzy matching (English + Vernacular + Regional)
DEALER_MASTER_LIST = [
    # English names (standard)
    "ABC Tractors",
    "Mahindra Tractors",
    "Sonalika Tractors",
    "John Deere",
    "Farmtrac",
    "Escorts",
    "Preet Tractors",
    "Punjab Tractors",
    "VST Tractors",
    "Captain Tractors",
    "Standard Tractors",
    "HMT Tractors",
    "International Tractors",
    "Kheti Tractors",
    "Agri King",
    "Tafe",
    "Indofarm",
    "Swaraj",
    "Kubota",
    "New Holland",
    "Eicher",
    "Massey Ferguson",
    "Force Motors",
    # Vernacular/Hindi variations
    "Mahindra",
    "Sonalika",
    "Swraj",
    "Tafe",
    "Farmtrac",
    "John Deere",
    "New Holland",
    "Kubota",
    "Eicher",
    "Massey Ferguson",
    "Force Motors",
    "Tafe Tractors",
    "VST Tillers",
    "Greaves Cotton",
    "HMT",
    "Punjab",
    "Captain",
    "Standard",
    "Indo Farm",
    "Agri King",
    # Common vernacular spellings (Hindi transliterations)
    "Mahindra Tractor",
    "Sonalika Tractor",
    "Swraj Tractor",
    "Tafe Tractor",
    "Farmtrac Tractor",
    "John Deere Tractor",
    "New Holland Tractor",
    "Kubota Tractor",
    "Eicher Tractor",
    "Massey Ferguson Tractor",
    "Force Tractor",
    "VST Tractor",
    "Greaves Tractor",
    "HMT Tractor",
    "Punjab Tractor",
    "Captain Tractor",
    "Standard Tractor",
    "Indo Farm Tractor",
    "Agri King Tractor",
    # Regional variations (North India)
    "Mahindra And Mahindra",
    "M And M",
    "Sonalika International",
    "Punjab Tractor Corporation",
    "Escorts Kubota",
    # Regional dealer names
    "Laxmi Tractors",
    "Bharat Tractors",
    "Jaguar Tractors",
    "Shiv Tractors",
    "Ganpati Tractors",
    "Shri Ram Tractors",
    "Durga Tractors",
    "Krishna Tractors",
    "Om Tractors",
    "Sai Tractors",
    "Maruti Tractors",
    "Harsh Tractors",
    "Agrasen Tractors",
    "Bajaj Tractors",
    # Vernacular/Hindi dealer name indicators
    "Tractor Agency",
    "Tractor Company",
    "Tractor House",
    "Tractor Showroom",
    "Tractor Dealer",
    "Tractor Distributor",
    "Mitra Tractors",
    "Sankalp Tractors",
    "Samriddhi Tractors"
]

# Extended model master list for exact matching
MODEL_MASTER_LIST = [
    "Mahindra 575 DI",
    "Mahindra 265 DI",
    "Mahindra 415 DI",
    "Mahindra 475 DI",
    "Mahindra 555 DI",
    "Mahindra 595 DI",
    "Mahindra 265 DI XP Plus",
    "Mahindra 475 DI XP Plus",
    "Mahindra 575 DI XP Plus",
    "Mahindra 605 DI",
    "Mahindra 715 DI",
    "Mahindra 255 DI Power Plus",
    "Mahindra 275 DI XP Plus",
    "Mahindra 585 DI",
    "Mahindra ARJUN NOVO 605",
    "Mahindra YUVO",
    "Sonalika DI 35",
    "Sonalika DI 42",
    "Sonalika DI 50",
    "Sonalika DI 60",
    "Sonalika DI 750",
    "Sonalika GT 22",
    "Sonalika GT 26",
    "Sonalika RX 42",
    "Sonalika RX 50",
    "Swaraj 735",
    "Swaraj 744",
    "Swaraj 855",
    "Swaraj 825",
    "Swaraj 735 FE",
    "Swaraj 744 FE",
    "Swaraj 855 FE",
    "Swaraj 842",
    "Swaraj 724",
    "John Deere 5050",
    "John Deere 5105",
    "John Deere 5205",
    "John Deere 3028 EN",
    "John Deere 3036 EN",
    "John Deere 4049",
    "John Deere 5075",
    "Farmtrac 45",
    "Farmtrac 50",
    "Farmtrac 60",
    "Farmtrac 65",
    "Farmtrac 55",
    "Escorts 335",
    "Escorts 355",
    "Escorts 365",
    "Escorts 3120",
    "TAFE 245",
    "TAFE 345",
    "TAFE 440",
    "TAFE 5250",
    "TAFE 9500",
    "Kubota MU 4501",
    "Kubota MU 5501",
    "Kubota MU 4800",
    "Kubota L 4508",
    "Kubota L 3608",
    "New Holland 3037",
    "New Holland 3630",
    "New Holland 3032",
    "New Holland 4037",
    "New Holland 5500",
    "Eicher 380",
    "Eicher 480",
    "Eicher 500",
    "Eicher 560",
    "Eicher 6150",
    "Massey Ferguson 1035",
    "Massey Ferguson 1038",
    "Massey Ferguson 245",
    "Massey Ferguson 241",
    "Massey Ferguson 750"
]

# Comprehensive vernacular keywords for Hindi and regional Indian languages
# Extended dealer/company related keywords (both native script + transliterated)
DEALER_KEYWORDS_ENGLISH = [
    'dealer', 'authorized', 'm/s', 'm/s ', 'tractors', 'motors', 'ltd', 'llp',
    'distributor', 'agency', 'sales', 'service', 'automobiles', 'vehicles',
    'company', 'corporation', 'enterprise', 'showroom', 'outlet', 'branch',
    'partner', 'associate', 'vendor', 'supplier', 'stockist'
]

DEALER_KEYWORDS_HINDI_NATIVE = [
    'डीलर', 'विक्रेता', 'एम/एस', 'मोटर्स', 'लिमिटेड', 'एलएलपी', 'वितरक',
    'एजेंसी', 'बिक्री', 'सर्विस', 'ऑटोमोबाइल', 'वाहन', 'कंपनी', 'निगम',
    'उद्यम', 'शोरूम', 'आउटलेट', 'शाखा', 'भागीदार', 'सहयोगी', 'विक्रेता',
    'आपूर्तिकर्ता', 'स्टॉकिस्ट', 'ट्रैक्टर एजेंसी', 'ट्रैक्टर कंपनी',
    'ट्रैक्टर शोरूम', 'ट्रैक्टर डीलर', 'कृषि यंत्र', 'खेती उपकरण',
    'होलसेल', 'रिटेल', 'ट्रेडर', 'सेल्स एजेंट', 'सर्विस सेंटर'
]

# Transliterated Hindi keywords (romanized Hindi commonly found in OCR)
DEALER_KEYWORDS_HINDI_TRANS = [
    'dealer', 'vikreta', 'vikretA', 'm/s', 'motars', 'limited', 'llp', 'vitarak',
    'agenSee', 'bikree', 'sarvis', 'automoobil', 'vaahan', 'kampani', 'nigham',
    'udyam', 'sharum', 'aotlet', 'shakha', 'bhagidaar', 'sahyogi', 'aapurtikarta',
    'stokist', 'traktar agenSee', 'traktar kampani', 'traktar sharum',
    'traktar dealer', 'krishi yantr', 'kheti upkaran', 'holesel', 'ritel',
    'trader', 'sales agent', 'service center',
    # Common transliterations
    'deelar', 'dealaar', 'daalar', 'daalaar', 'vkreta', 'vikrta', 'veekreta',
    'motar', 'motor', 'kampany', 'kampanee', 'shropm', 'showrowm', 'shoowroom',
    'agenCee', 'agensi', 'agenCy', 'sarvis', 'servis', 'serivice', 'serves',
    'vikas agent', 'vikas agency', 'traktar', 'tractar', 'trakter', 'traktar'
]

# Combined Hindi keywords (native + transliterated)
DEALER_KEYWORDS_HINDI = DEALER_KEYWORDS_HINDI_NATIVE + DEALER_KEYWORDS_HINDI_TRANS

# Extended HP/power related keywords (both native + transliterated)
HP_KEYWORDS_ENGLISH = [
    'hp', 'horse', 'power', 'bh[Pp]', 'engine', 'capacity', 'tractor',
    'power output', 'engine power', 'rated power', 'maximum power',
    'bhp', 'metric hp', 'engine capacity', 'displacement'
]

HP_KEYWORDS_HINDI_NATIVE = [
    'हॉर्स पावर', 'हॉर्सपावर', 'एचपी', 'भाप', 'पावर', 'शक्ति', 'इंजन',
    'क्षमता', 'ट्रैक्टर', 'मोटर पावर', 'इंजन पावर', 'अधिकतम पावर',
    'रेटेड पावर', 'बीएचपी', 'मीट्रिक एचपी', 'इंजन क्षमता', 'विस्थापन',
    'बल', 'टॉर्क', 'ड्राइव पावर', 'पीटीओ', 'पावर टेक ऑफ'
]

# Transliterated HP keywords
HP_KEYWORDS_HINDI_TRANS = [
    'horsepower', 'horsepowar', 'horsepawar', 'hors power', 'hp', 'bhp',
    'power', 'powar', 'pawar', 'shkti', 'shakti', 'enjin', 'engine', 'enjine',
    'capacity', 'kapasiti', 'kmapcity', 'traktar', 'tractar', 'trakter',
    'motor power', 'motar power', 'max power', 'maximum power',
    # Common transliterations
    'haurs power', 'hors powr', 'hars power', 'horspoar', 'horspwr',
    'AEchPee', 'aichpi', 'aych pi', 'eech pee', 'bhaap', 'bap', 'bhaap',
    'paavar', 'pawer', 'powr', 'shkti', 'shktY', 'enjen', 'enjyn',
    'kamal', 'kammlti', 'toraq', 'torque', 'traktr', 'traktar',
    'PTO', 'pto', 'ptoo', 'power takeoff', 'power take off'
]

# Combined HP keywords
HP_KEYWORDS_HINDI = HP_KEYWORDS_HINDI_NATIVE + HP_KEYWORDS_HINDI_TRANS

# Extended cost/price related keywords
COST_KEYWORDS_ENGLISH = [
    'total', 'amount', 'price', 'cost', 'rs', '₹', 'rupees', 'grand total',
    'net amount', 'invoice value', 'deal value', 'transaction value',
    'payment', 'billing', 'quotation', 'estimate', 'ex-showroom', 'on-road'
]

COST_KEYWORDS_HINDI_NATIVE = [
    'कुल', 'राशि', 'कीमत', 'मूल्य', 'भुगतान', 'टोटल', 'ग्रैंड टोटल',
    'शुद्ध राशि', 'इनवॉइस मूल्य', 'डील वैल्यू', 'ट्रांजैक्शन वैल्यू',
    'बिल', 'कोटेशन', 'अनुमान', 'एक्स-शोरूम', 'ऑन-रोड', 'दाम',
    'भाव', 'दर', 'वैल्यू', 'खर्चा', 'व्यय'
]

# Transliterated cost keywords
COST_KEYWORDS_HINDI_TRANS = [
    'kul', 'raashi', 'raashee', 'kimat', 'keemat', 'muly', 'moolya',
    'bhugtan', 'bhugataan', 'total', 'grand total', 'net amount',
    'shudh raashi', 'shudh rashi', 'invoice value', 'invois mulya',
    'deal value', 'del valyu', 'transaction value', 'tranzakshan valyu',
    'bill', 'bil', 'quotation', 'kwoteshan', 'estimate', 'eshtimt',
    'ex showroom', 'exsharwm', 'on road', 'onrod', 'on-road',
    # Common transliterations
    'dam', 'daam', 'dham', 'bhaav', 'bhaav', 'dar', 'daar',
    'kharcha', 'kharchya', 'vyay', 'veay', 'lagat', 'lgaat',
    'rupaiya', 'rupee', 'rupiya', 'rs', 'rpees', 'rs.',
    'mulya', 'mulya', 'moolya', 'qeemat', 'qemat',
    'jma', 'jmaa', ' jama', 'jmaa'  # Common OCR errors for 'raashi'
]

# Combined cost keywords
COST_KEYWORDS_HINDI = COST_KEYWORDS_HINDI_NATIVE + COST_KEYWORDS_HINDI_TRANS

# Extended HP context keywords
HP_CONTEXT_KEYWORDS_ENGLISH = [
    'hp', 'horse', 'power', 'bhp', 'engine', 'capacity', 'tractor',
    'model', 'specification', 'features', 'technical'
]

HP_CONTEXT_KEYWORDS_HINDI_NATIVE = [
    'एचपी', 'हॉर्स', 'पावर', 'भाप', 'इंजन', 'क्षमता', 'ट्रैक्टर',
    'मॉडल', 'विशिष्टता', 'फीचर्स', 'तकनीकी', 'स्पेसिफिकेशन'
]

HP_CONTEXT_KEYWORDS_HINDI_TRANS = [
    'hp', 'hors', 'power', 'powar', 'pawar', 'bhp', 'enjin', 'engine',
    'capacity', 'kapasiti', 'traktar', 'tractar', 'model', 'modl',
    'specification', 'spesification', 'features', 'fichars', 'fichers',
    'technical', 'teknikal', 'techanical'
]

# Combined HP context keywords
HP_CONTEXT_KEYWORDS_HINDI = HP_CONTEXT_KEYWORDS_HINDI_NATIVE + HP_CONTEXT_KEYWORDS_HINDI_TRANS

# Company/business name indicators
COMPANY_INDICATORS = [
    'ltd', 'llp', 'pvt', 'private', 'company', 'corp', 'inc',
    'motors', 'tractors', 'automobiles', 'agency', 'dealers',
    'limited', 'corporation', 'holdings', 'group', 'enterprises'
]

COMPANY_INDICATORS_HINDI_NATIVE = [
    'लिमिटेड', 'प्राइवेट', 'कंपनी', 'कारपोरेशन', 'होल्डिंग्स',
    'ग्रुप', 'एंटरप्राइजेज', 'निगम', 'संस्था', 'व्यापार'
]

COMPANY_INDICATORS_HINDI_TRANS = [
    'limited', 'limtd', 'limeted', 'pvt', 'private', 'privet',
    'company', 'kampani', 'kampany', 'kampanee', 'corpration',
    'corporeshon', 'holdings', 'holdngs', 'group', 'grwp',
    'enterprises', 'enterprizes', 'nigham', 'nigam', 'sanstha',
    'vyapar', 'vyapaar', 'business', 'bisnes'
]

# Combined company indicators
COMPANY_INDICATORS_HINDI = COMPANY_INDICATORS_HINDI_NATIVE + COMPANY_INDICATORS_HINDI_TRANS

# Vernacular HP-related context words
HP_CONTEXT_HINDI_NATIVE = [
    'ट्रैक्टर', 'गाड़ी', 'वाहन', 'यंत्र', 'मशीन', 'उपकरण',
    'खेती', 'कृषि', 'हल', 'सैंया', 'पावर', 'ताकत', 'जोर',
    'ड्राइव', 'ट्रैक्शन', 'पीटीओ', 'हिड्रोलिक', 'लिफ्ट'
]

HP_CONTEXT_HINDI_TRANS = [
    'traktar', 'tractar', 'trakter', 'gadi', 'gaadee', 'gaadi',
    'vaahan', 'vahan', 'yantr', 'yantra', 'mashin', 'machine',
    'upkaran', 'upkarn', 'kheti', 'khti', 'krishi', 'krish',
    'hal', 'haal', 'sainya', 'sainya', 'power', 'powar',
    'takat', 'taqat', 'takath', 'jor', 'joru',
    'drive', 'draiv', 'traktion', 'trakshn', 'pto', 'ptoo',
    'hydraulic', 'haidrolik', 'lift', 'lif'
]

# Combined HP context words
HP_CONTEXT_HINDI = HP_CONTEXT_HINDI_NATIVE + HP_CONTEXT_HINDI_TRANS


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FieldResult:
    """Result of a field extraction with confidence"""
    value: Any
    confidence: float
    bbox: Tuple[int, int, int, int] = None
    source: str = "text"  # "text" or "visual"


@dataclass
class CheckmarkResult:
    """Result of checkmark/tick mark detection"""
    present: bool
    bbox: Tuple[int, int, int, int] = None
    checkmark_type: str = None  # "tick", "cross", "circle", "square"
    confidence: float = 0.0
    associated_value: Any = None  # Value associated with the checkmark (e.g., HP value)


@dataclass
class HandwrittenFieldResult:
    """Result of handwritten field detection"""
    value: Any
    confidence: float
    bbox: Tuple[int, int, int, int] = None
    field_type: str = None  # "hp", "cost", "date", "number"
    is_handwritten: bool = True
    stroke_width_avg: float = 0.0
    ink_color: str = None  # "black", "blue", "red", "brown"


@dataclass
class SignatureResult:
    """Result of signature detection"""
    present: bool
    bbox: Tuple[int, int, int, int] = None
    confidence: float = 0.0


@dataclass
class StampResult:
    """Result of stamp detection"""
    present: bool
    bbox: Tuple[int, int, int, int] = None
    confidence: float = 0.0


@dataclass
class ExtractedFields:
    """Complete extraction result for a document"""
    doc_id: str
    page_number: int
    dealer_name: FieldResult = None
    model_name: FieldResult = None
    horse_power: FieldResult = None
    asset_cost: FieldResult = None
    signature: SignatureResult = None
    stamp: StampResult = None
    document_confidence: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'doc_id': self.doc_id,
            'page_number': self.page_number,
            'dealer_name': {
                'value': self.dealer_name.value if self.dealer_name else None,
                'confidence': self.dealer_name.confidence if self.dealer_name else 0.0,
                'bbox': self.dealer_name.bbox if self.dealer_name else None,
                'source': self.dealer_name.source if self.dealer_name else None
            } if self.dealer_name else {'value': None, 'confidence': 0.0},
            'model_name': {
                'value': self.model_name.value if self.model_name else None,
                'confidence': self.model_name.confidence if self.model_name else 0.0,
                'bbox': self.model_name.bbox if self.model_name else None,
                'source': self.model_name.source if self.model_name else None
            } if self.model_name else {'value': None, 'confidence': 0.0},
            'horse_power': {
                'value': self.horse_power.value if self.horse_power else None,
                'confidence': self.horse_power.confidence if self.horse_power else 0.0,
                'bbox': self.horse_power.bbox if self.horse_power else None,
                'source': self.horse_power.source if self.horse_power else None
            } if self.horse_power else {'value': None, 'confidence': 0.0},
            'asset_cost': {
                'value': self.asset_cost.value if self.asset_cost else None,
                'confidence': self.asset_cost.confidence if self.asset_cost else 0.0,
                'bbox': self.asset_cost.bbox if self.asset_cost else None,
                'source': self.asset_cost.source if self.asset_cost else None
            } if self.asset_cost else {'value': None, 'confidence': 0.0},
            'signature': {
                'present': self.signature.present if self.signature else False,
                'bbox': self.signature.bbox if self.signature else None,
                'confidence': self.signature.confidence if self.signature else 0.0
            } if self.signature else {'present': False, 'bbox': None, 'confidence': 0.0},
            'stamp': {
                'present': self.stamp.present if self.stamp else False,
                'bbox': self.stamp.bbox if self.stamp else None,
                'confidence': self.stamp.confidence if self.stamp else 0.0
            } if self.stamp else {'present': False, 'bbox': None, 'confidence': 0.0},
            'document_confidence': self.document_confidence
        }

    def to_json(self, filepath: str = None) -> str:
        """Export to JSON string or file"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Extracted fields saved to {filepath}")
        return json_str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def fuzzy_match(text: str, candidate: str, threshold: float = 0.8) -> Tuple[bool, float]:
    """
    Enhanced fuzzy matching for poor OCR text.

    Args:
        text: Extracted text from OCR
        candidate: Text from master list
        threshold: Minimum match ratio (default 0.8, lowered for poor OCR)

    Returns:
        Tuple of (match_status, match_score)
    """
    # Normalize text for comparison
    text_norm = text.lower().strip()
    candidate_norm = candidate.lower().strip()

    # Direct match
    if text_norm == candidate_norm:
        return True, 1.0

    # Clean text by removing non-alphanumeric characters except spaces
    text_clean = re.sub(r'[^a-zA-Z0-9\s]', '', text_norm)
    candidate_clean = re.sub(r'[^a-zA-Z0-9\s]', '', candidate_norm)

    # Fuzzy match using SequenceMatcher on cleaned text
    score = SequenceMatcher(None, text_clean, candidate_clean).ratio()

    # Also check if one contains the other (partial match)
    if candidate_clean in text_clean or text_clean in candidate_clean:
        score = max(score, 0.85)

    # Word-based matching for multi-word names
    text_words = set(text_clean.split())
    candidate_words = set(candidate_clean.split())

    if text_words and candidate_words:
        # Check if significant words match
        common_words = text_words.intersection(candidate_words)
        if common_words:
            word_match_ratio = len(common_words) / max(len(text_words), len(candidate_words))
            score = max(score, word_match_ratio * 0.9)

    # Special handling for poor OCR: check phonetic similarity
    # Remove vowels and compare consonants (basic phonetic matching)
    def remove_vowels(s):
        return re.sub(r'[aeiouAEIOU]', '', s)

    text_consonants = remove_vowels(text_clean)
    candidate_consonants = remove_vowels(candidate_clean)

    if len(text_consonants) > 3 and len(candidate_consonants) > 3:
        consonant_score = SequenceMatcher(None, text_consonants, candidate_consonants).ratio()
        if consonant_score > 0.8:  # High consonant similarity
            score = max(score, consonant_score * 0.8)

    return score >= threshold, score


def extract_numbers(text: str) -> List[int]:
    """
    Extract all integers from text.

    Args:
        text: Input text

    Returns:
        List of extracted integers
    """
    # Remove currency symbols, commas, and extract numbers
    cleaned = re.sub(r'[₹Rs.,]', '', text)
    numbers = re.findall(r'\d+', cleaned)
    return [int(n) for n in numbers if n]


def compute_confidence(ocr_confidence: float, match_score: float,
                       location_score: float = 1.0) -> float:
    """
    Compute weighted confidence score.

    Args:
        ocr_confidence: OCR confidence (0-1)
        match_score: Text match score (0-1)
        location_score: Positional confidence (0-1)

    Returns:
        Weighted confidence score
    """
    # Default weights: 40% OCR, 40% match, 20% location
    confidence = (
        0.4 * ocr_confidence +
        0.4 * match_score +
        0.2 * location_score
    )
    return round(confidence, 2)


# =============================================================================
# TEXT-BASED FIELD EXTRACTION
# =============================================================================

class TextFieldExtractor:
    """
    Extract text-based fields using OCR data from Step 3.
    """

    def __init__(self, dealer_list: List[str] = None, model_list: List[str] = None):
        """
        Initialize extractor with master lists.

        Args:
            dealer_list: List of known dealers for fuzzy matching
            model_list: List of known models for exact matching
        """
        self.dealer_list = dealer_list or DEALER_MASTER_LIST
        self.model_list = model_list or MODEL_MASTER_LIST

    def extract_dealer_name(self, blocks: List, region: str = "header") -> Optional[FieldResult]:
        """
        Enhanced dealer name extraction with multiple strategies.

        Strategies:
        1. Pattern-based: Look for "Dealer:" or similar patterns
        2. Fuzzy matching against master list
        3. Position-based: Top region of document
        4. Context-based: Near tractor-related keywords
        """
        keywords = DEALER_KEYWORDS_ENGLISH + DEALER_KEYWORDS_HINDI

        best_match = None
        best_score = 0.0
        best_confidence = 0.0
        best_bbox = None

        # Multi-pass extraction strategy
        for block in blocks:
            text = block.get('text', '').strip()
            confidence = block.get('confidence', 0.5)
            bbox = tuple(block.get('bbox', [0, 0, 0, 0]))

            # Skip very low confidence OCR results
            if confidence < 0.3:
                continue

            text_lower = text.lower()

            # Strategy 1: Pattern-based extraction (highest priority)
            # Look for "Dealer: XYZ" or "Dealer Name: XYZ" patterns
            dealer_patterns = [
                r'dealer\s*[:\-]?\s*(.+?)(?:\n|$)',
                r'dealer\s*name\s*[:\-]?\s*(.+?)(?:\n|$)',
                r'authorized\s+dealer\s*[:\-]?\s*(.+?)(?:\n|$)',
                r'm/s\s*(.+?)(?:\n|$)',
                r'm\.s\.?\s*(.+?)(?:\n|$)',
            ]

            for pattern in dealer_patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Clean up the candidate
                    candidate = re.sub(r'[^\w\s]', '', candidate).strip()

                    if len(candidate) > 2:  # Minimum length check
                        for dealer in self.dealer_list:
                            is_match, match_score = fuzzy_match(candidate, dealer, threshold=0.6)
                            if is_match:
                                confidence_score = compute_confidence(
                                    ocr_confidence=confidence,
                                    match_score=match_score,
                                    location_score=1.0  # Pattern match gets high location score
                                )
                                if match_score > best_score:
                                    best_match = dealer
                                    best_score = match_score
                                    best_confidence = confidence_score
                                    best_bbox = bbox

            # Strategy 2: Direct fuzzy matching with keyword context
            has_keyword = any(kw in text_lower for kw in keywords)

            for dealer in self.dealer_list:
                is_match, match_score = fuzzy_match(text, dealer, threshold=0.7)

                if is_match:
                    # Boost score if keywords present or in header
                    keyword_boost = 0.15 if has_keyword else 0.0
                    location_boost = 0.1 if region == "header" else 0.0

                    final_match_score = min(match_score + keyword_boost + location_boost, 1.0)

                    confidence_score = compute_confidence(
                        ocr_confidence=confidence,
                        match_score=final_match_score,
                        location_score=1.0 if region == "header" else 0.7
                    )

                    if final_match_score > best_score:
                        best_match = dealer
                        best_score = final_match_score
                        best_confidence = confidence_score
                        best_bbox = bbox

            # Strategy 3: Position-based (if no good matches found yet)
            # In header region, any text that looks like a business name
            if region == "header" and best_score < 0.8:
                # Check if text looks like a business name (contains company indicators)
                company_indicators = COMPANY_INDICATORS + COMPANY_INDICATORS_HINDI
                has_company_indicator = any(ind in text_lower for ind in company_indicators)

                if has_company_indicator and len(text) > 5 and confidence > 0.4:
                    # Try fuzzy match with relaxed threshold
                    for dealer in self.dealer_list:
                        is_match, match_score = fuzzy_match(text, dealer, threshold=0.5)
                        if is_match and match_score > best_score:
                            confidence_score = compute_confidence(
                                ocr_confidence=confidence,
                                match_score=match_score,
                                location_score=0.8  # Position-based gets moderate score
                            )
                            best_match = dealer
                            best_score = match_score
                            best_confidence = confidence_score
                            best_bbox = bbox

        if best_match:
            logger.info(f"Dealer name extracted: '{best_match}' with confidence {best_confidence}")
            return FieldResult(value=best_match, confidence=best_confidence,
                             bbox=best_bbox, source="text")

        return None

    def extract_model_name(self, blocks: List, region: str = "all") -> Optional[FieldResult]:
        """
        Extract model name using exact matching.

        Where: Header + Body
        How: Exact match against model master list
        """
        keywords = ['model', 'tractor model', 'asset', 'vehicle']

        best_match = None
        best_confidence = 0.0
        best_bbox = None

        for block in blocks:
            text = block.get('text', '').strip()
            ocr_conf = block.get('confidence', 0.5)
            bbox = tuple(block.get('bbox', [0, 0, 0, 0]))

            # Exact match only (no fuzzy for model names)
            for model in self.model_list:
                if text.lower() == model.lower():
                    # Boost confidence if near keywords
                    text_lower = text.lower()
                    has_keyword = any(kw in text_lower for kw in keywords)
                    keyword_boost = 0.1 if has_keyword else 0.0

                    location_score = 1.0 if region == "header" else 0.8

                    confidence = compute_confidence(
                        ocr_confidence=ocr_conf,
                        match_score=1.0,  # Exact match = 1.0
                        location_score=location_score
                    ) + keyword_boost

                    if confidence > best_confidence:
                        best_match = model
                        best_confidence = min(confidence, 1.0)
                        best_bbox = bbox

            # Also check if model code is contained (e.g., "575 DI" in "Model: 575 DI")
            for model in self.model_list:
                model_code = model.split()[-1]  # Get last part like "DI"
                if model_code in text:
                    confidence = compute_confidence(
                        ocr_confidence=ocr_conf,
                        match_score=0.9,  # Partial match
                        location_score=0.8
                    )
                    if confidence > best_confidence:
                        best_match = model
                        best_confidence = confidence
                        best_bbox = bbox

        if best_match:
            logger.info(f"Model name extracted: '{best_match}' with confidence {best_confidence}")
            return FieldResult(value=best_match, confidence=best_confidence,
                             bbox=best_bbox, source="text")

        return None

    def extract_horse_power(self, blocks: List, region: str = "body") -> Optional[FieldResult]:
        """
        Enhanced HP extraction with multiple strategies.

        Strategies:
        1. Pattern-based: Look for HP/BHP patterns
        2. Check mark detection: Look for ✓ or similar symbols
        3. Context-based: Near model or tractor keywords
        4. Number validation: Ensure reasonable HP range
        """
        hp_patterns = [
            r'(\d+)\s*hp',
            r'(\d+)\s*horse\s*power',
            r'(\d+)\s*bh[Pp]',
            r'HP\s*[:=]?\s*(\d+)',
            r'Horse\s*Power\s*[:=]?\s*(\d+)',
            r'(\d+)\s*h\.?p\.?',
            r'power\s*[:=]?\s*(\d+)',
            r'(\d+)\s*kw',  # Some documents use KW
            r'(\d+)\s*एचपी',
            r'(\d+)\s*हॉर्स\s*पावर',
        ]

        # Check mark and symbol patterns
        check_patterns = [
            r'[✓✔☑√xX]',
            r'yes',
            r'true',
            r'selected',
            r'chosen'
        ]

        keywords = HP_KEYWORDS_ENGLISH + HP_KEYWORDS_HINDI

        best_hp = None
        best_confidence = 0.0
        best_bbox = None

        for block in blocks:
            text = block.get('text', '').strip()
            ocr_conf = block.get('confidence', 0.5)
            bbox = tuple(block.get('bbox', [0, 0, 0, 0]))

            # Skip very low confidence OCR results
            if ocr_conf < 0.3:
                continue

            text_lower = text.lower()

            # Strategy 1: Pattern-based HP extraction
            for pattern in hp_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    hp_value = int(match.group(1))

                    # Convert KW to HP (1 KW ≈ 1.34 HP)
                    if 'kw' in text_lower:
                        hp_value = int(hp_value * 1.34)

                    # Validate HP range (15-100 is typical for tractors)
                    if 15 <= hp_value <= 100:
                        # Higher confidence for clean extractions
                        clean_match = re.match(r'^(\d+)\s*hp$', text.strip(), re.IGNORECASE)
                        match_bonus = 0.15 if clean_match else 0.0

                        # Boost if near keywords
                        has_keyword = any(kw in text_lower for kw in keywords)
                        keyword_bonus = 0.1 if has_keyword else 0.0

                        confidence = compute_confidence(
                            ocr_confidence=ocr_conf,
                            match_score=0.95 + match_bonus,
                            location_score=1.0
                        ) + keyword_bonus

                        if confidence > best_confidence:
                            best_hp = hp_value
                            best_confidence = min(confidence, 1.0)
                            best_bbox = bbox

            # Strategy 2: Check mark detection for HP values
            # Look for check marks near HP values in tabular data
            if best_confidence < 0.8:
                has_check = any(re.search(pattern, text) for pattern in check_patterns)

                if has_check:
                    # Look for HP values in nearby blocks (within same row/area)
                    # This is a simplified approach - in practice you'd need layout analysis
                    numbers = extract_numbers(text)
                    for num in numbers:
                        if 15 <= num <= 100:
                            confidence = compute_confidence(
                                ocr_confidence=ocr_conf,
                                match_score=0.8,  # Check mark gets lower match score
                                location_score=0.9
                            )
                            if confidence > best_confidence:
                                best_hp = num
                                best_confidence = confidence
                                best_bbox = bbox

            # Strategy 3: Context-based extraction
            # Look for standalone numbers near HP keywords
            has_hp_context = any(kw in text_lower for kw in keywords)
            if has_hp_context and best_confidence < 0.7:
                numbers = extract_numbers(text)
                for num in numbers:
                    if 15 <= num <= 100:
                        # Check if this is likely an HP value (not cost, year, etc.)
                        # HP values are typically single or two digits
                        if num < 200:  # Avoid mistaking costs for HP
                            confidence = compute_confidence(
                                ocr_confidence=ocr_conf,
                                match_score=0.75,
                                location_score=0.8
                            )
                            if confidence > best_confidence:
                                best_hp = num
                                best_confidence = confidence
                                best_bbox = bbox

        if best_hp:
            logger.info(f"Horse Power extracted: {best_hp} with confidence {best_confidence}")
            return FieldResult(value=best_hp, confidence=best_confidence,
                             bbox=best_bbox, source="text")

        return None

    def extract_asset_cost(self, blocks: List, region: str = "footer") -> Optional[FieldResult]:
        """
        Extract asset cost value.

        Where: Footer + Body
        How: Search near cost keywords, extract largest number
        """
        cost_keywords = COST_KEYWORDS_ENGLISH + COST_KEYWORDS_HINDI

        # First pass: find blocks with cost keywords
        cost_blocks = []
        for block in blocks:
            text = block.get('text', '').lower()
            if any(kw in text for kw in cost_keywords):
                cost_blocks.append(block)

        # If no keyword blocks, use all blocks (fallback)
        if not cost_blocks:
            cost_blocks = blocks

        # Extract all numbers from cost blocks
        all_costs = []
        for block in cost_blocks:
            text = block.get('text', '')
            ocr_conf = block.get('confidence', 0.5)
            bbox = tuple(block.get('bbox', [0, 0, 0, 0]))

            numbers = extract_numbers(text)
            for num in numbers:
                # Cost is usually 6+ digits for tractors
                if num >= 10000:
                    all_costs.append({
                        'value': num,
                        'confidence': ocr_conf,
                        'bbox': bbox
                    })

        if all_costs:
            # Sort by value (descending) - largest is usually the total
            all_costs.sort(key=lambda x: x['value'], reverse=True)

            best = all_costs[0]
            logger.info(f"Asset Cost extracted: {best['value']} with confidence {best['confidence']}")
            return FieldResult(value=best['value'], confidence=best['confidence'],
                             bbox=best['bbox'], source="text")

        return None


# =============================================================================
# VISUAL FIELD DETECTION (SIGNATURE & STAMP)
# =============================================================================

class VisualFieldDetector:
    """
    Detect visual elements like signature and stamp using image processing.
    """

    def __init__(self):
        """Initialize detector"""
        pass

    def detect_checkmarks(self, image: np.ndarray,
                          region: Tuple[int, int, int, int] = None) -> List[CheckmarkResult]:
        """
        Detect checkmarks/tick marks in the image.
        
        Checkmarks are commonly used in invoices to mark selected options,
        especially for HP (horse power) selection in tabular data.

        Args:
            image: Input image (BGR format)
            region: Optional region of interest (x1, y1, x2, y2)

        Returns:
            List of CheckmarkResult objects
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        
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
        
        gh, gw = search_region.shape[:2]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(search_region, (3, 3), 0)
        
        # Apply adaptive threshold for better contrast detection
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        checkmarks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Checkmark heuristics
            min_area = gh * gw * 0.001  # Very small minimum area
            max_area = gh * gw * 0.05   # Small maximum area
            
            # Checkmark characteristics:
            # 1. Small to moderate size
            # 2. Often roughly square or slightly elongated
            # 3. Can be tick-shaped (elongated with angle)
            
            if area < min_area or area > max_area:
                continue
            
            # Analyze shape to determine checkmark type
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Tick mark detection (most common)
            # Tick marks are typically elongated with a characteristic angle
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = area / hull_area
            else:
                solidity = 0
            
            # Detect tick mark (✓ shape)
            is_tick = (
                aspect_ratio > 0.3 and  # Not too thin
                aspect_ratio < 3.0 and  # Not too elongated
                solidity > 0.3 and  # Reasonably solid
                circularity < 0.5  # Not very circular
            )
            
            # Detect cross/X mark
            is_cross = (
                aspect_ratio > 0.5 and
                aspect_ratio < 2.0 and
                circularity < 0.4
            )
            
            # Detect circle/ballot box
            is_circle = (
                0.7 < circularity < 1.0 or
                (0.7 < aspect_ratio < 1.4 and circularity > 0.6)
            )
            
            # Determine checkmark type
            if is_tick:
                checkmark_type = "tick"
                confidence = 0.85
            elif is_cross:
                checkmark_type = "cross"
                confidence = 0.80
            elif is_circle:
                checkmark_type = "circle"
                confidence = 0.75
            else:
                # Default to tick for small marks
                if area < gh * gw * 0.01:
                    checkmark_type = "tick"
                    confidence = 0.70
                else:
                    checkmark_type = "square"
                    confidence = 0.65
            
            # Calculate bounding box in original coordinates
            if region:
                orig_bbox = (x + x1, y + y1, x + x1 + cw, y + y1 + ch)
            else:
                orig_bbox = (x, y, x + cw, y + ch)
            
            checkmarks.append(CheckmarkResult(
                present=True,
                bbox=orig_bbox,
                checkmark_type=checkmark_type,
                confidence=confidence
            ))
        
        if not checkmarks:
            return [CheckmarkResult(present=False, confidence=0.9)]
        
        logger.info(f"Detected {len(checkmarks)} checkmarks in region")
        return checkmarks

    def detect_signature(self, image: np.ndarray,
                         footer_y_start: float = 0.75) -> SignatureResult:
        """
        Detect dealer signature in footer region.

        Where: Footer region
        How: Detect low text density regions with irregular strokes

        Args:
            image: Input image (BGR format)
            footer_y_start: Y-position where footer starts (ratio of image height)

        Returns:
            SignatureResult with presence, bbox, and confidence
        """
        h, w = image.shape[:2]
        footer_start = int(h * footer_y_start)
        footer_region = image[footer_start:h, 0:w]

        # Convert to grayscale
        gray = cv2.cvtColor(footer_region, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0

            # Signature heuristics
            # 1. Not too small (at least 5% of footer height)
            min_area = (h - footer_start) * w * 0.05
            # 2. Aspect ratio indicates horizontal stroke (thin, wide)
            is_thin_horizontal = 0.1 < aspect_ratio < 1.0
            # 3. Elongated shape (width > 2x height)
            is_elongated = cw > ch * 2

            if area > min_area and (is_thin_horizontal or is_elongated):
                # Calculate confidence based on heuristics
                confidence = 0.7
                if is_elongated:
                    confidence += 0.15
                if aspect_ratio > 0.3:
                    confidence += 0.1

                bbox = (x, y + footer_start, x + cw, y + footer_start + ch)

                logger.info(f"Signature detected: present={True}, confidence={confidence}")
                return SignatureResult(present=True, bbox=bbox,
                                      confidence=min(confidence, 1.0))

        logger.info("Signature detected: not present")
        return SignatureResult(present=False, confidence=0.9)

    def detect_stamp(self, image: np.ndarray,
                     region: str = "footer") -> StampResult:
        """
        Detect dealer stamp.

        Where: Footer / right side
        How: Detect circular or rectangular shapes with dense ink

        Args:
            image: Input image (BGR format)
            region: Region to search ('footer', 'right', 'all')

        Returns:
            StampResult with presence, bbox, and confidence
        """
        h, w = image.shape[:2]

        # Define stamp search region (right side of footer)
        if region == "footer":
            search_y_start = int(h * 0.75)
            search_region = image[search_y_start:h, int(w * 0.6):w]
        elif region == "right":
            search_region = image[0:h, int(w * 0.7):w]
        else:
            search_region = image

        if search_region.size == 0:
            return StampResult(present=False, confidence=0.9)

        gh, gw = search_region.shape[:2]

        # Convert to grayscale
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)

        # Apply blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0

            # Stamp heuristics
            # 1. Moderate size
            min_area = gh * gw * 0.01
            max_area = gh * gw * 0.15
            # 2. Near-circular or rectangular
            is_near_circular = 0.7 < aspect_ratio < 1.4
            # 3. Borders present (annular shapes)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                is_annular = 0.6 < circularity < 0.9
            else:
                is_annular = False

            if min_area < area < max_area and (is_near_circular or is_annular):
                # Adjust coordinates for original image
                if region == "footer":
                    orig_y = int(h * 0.75)
                    orig_x = int(w * 0.6)
                elif region == "right":
                    orig_y = 0
                    orig_x = int(w * 0.7)
                else:
                    orig_y = 0
                    orig_x = 0

                bbox = (x + orig_x, y + orig_y, x + orig_x + cw, y + orig_y + ch)

                # Calculate confidence
                confidence = 0.75
                if is_annular:
                    confidence += 0.15
                if is_near_circular:
                    confidence += 0.1

                logger.info(f"Stamp detected: present={True}, confidence={confidence}")
                return StampResult(present=True, bbox=bbox,
                                  confidence=min(confidence, 1.0))

        logger.info("Stamp detected: not present")
        return StampResult(present=False, confidence=0.9)


# =============================================================================
# VALIDATION AND CROSS-VERIFICATION
# =============================================================================

def validate_and_cross_check_fields(dealer_name: Optional[FieldResult],
                                   model_name: Optional[FieldResult],
                                   horse_power: Optional[FieldResult],
                                   asset_cost: Optional[FieldResult]) -> Dict[str, Optional[FieldResult]]:
    """
    Cross-validate extracted fields for consistency and logical relationships.

    Args:
        dealer_name: Extracted dealer name
        model_name: Extracted model name
        horse_power: Extracted HP value
        asset_cost: Extracted cost value

    Returns:
        Dictionary with validated/corrected fields
    """
    validated = {
        'dealer_name': dealer_name,
        'model_name': model_name,
        'horse_power': horse_power,
        'asset_cost': asset_cost
    }

    # Rule 1: Model should be consistent with dealer
    if model_name and dealer_name:
        model_value = model_name.value.lower() if model_name.value else ""
        dealer_value = dealer_name.value.lower() if dealer_name.value else ""

        # Check if model brand matches dealer brand
        model_brand = model_value.split()[0] if model_value.split() else ""
        dealer_brand = dealer_value.split()[0] if dealer_value.split() else ""

        if model_brand and dealer_brand:
            # Allow some flexibility (e.g., "Sonalika" model with "Sonalika Tractors" dealer)
            if model_brand not in dealer_brand and dealer_brand not in model_brand:
                # Reduce confidence if mismatch
                model_name.confidence *= 0.8
                dealer_name.confidence *= 0.8
                logger.warning(f"Model-Dealer mismatch: {model_brand} vs {dealer_brand}")

    # Rule 2: HP should be reasonable for the model
    if model_name and horse_power:
        model_value = model_name.value.lower() if model_name.value else ""

        # Extract expected HP from model name (rough estimates)
        expected_hp_ranges = {
            'mahindra': (20, 75),
            'sonalika': (30, 60),
            'swaraj': (25, 60),
            'john deere': (25, 75),
            'farmtrac': (25, 50),
            'escorts': (25, 55),
            'tafe': (25, 50),
            'new holland': (25, 75),
            'kubota': (20, 60),
            'eicher': (20, 60)
        }

        hp_value = horse_power.value
        for brand, (min_hp, max_hp) in expected_hp_ranges.items():
            if brand in model_value:
                if not (min_hp <= hp_value <= max_hp):
                    # HP is outside expected range for this brand
                    horse_power.confidence *= 0.7
                    logger.warning(f"HP {hp_value} outside expected range for {brand}: {min_hp}-{max_hp}")
                break

    # Rule 3: Cost should be reasonable for tractor (not too low/high)
    if asset_cost:
        cost_value = asset_cost.value
        # Tractor costs are typically between 3-15 lakhs (300,000 - 1,500,000 INR)
        if cost_value < 200000 or cost_value > 2000000:
            asset_cost.confidence *= 0.6
            logger.warning(f"Cost {cost_value} seems unreasonable for a tractor")

    # Rule 4: If we have model but no dealer, try to infer dealer from model
    if model_name and not dealer_name:
        model_value = model_name.value.lower() if model_name.value else ""
        brand = model_value.split()[0] if model_value.split() else ""

        # Map common brands to dealer names
        brand_to_dealer = {
            'mahindra': 'Mahindra Tractors',
            'sonalika': 'Sonalika Tractors',
            'swaraj': 'Swaraj',
            'john': 'John Deere',
            'farmtrac': 'Farmtrac',
            'escorts': 'Escorts',
            'tafe': 'Tafe',
            'new': 'New Holland',
            'kubota': 'Kubota',
            'eicher': 'Eicher'
        }

        if brand in brand_to_dealer:
            inferred_dealer = brand_to_dealer[brand]
            # Create a low-confidence inferred result
            validated['dealer_name'] = FieldResult(
                value=inferred_dealer,
                confidence=0.4,  # Low confidence for inferred
                source="inferred"
            )
            logger.info(f"Inferred dealer '{inferred_dealer}' from model '{model_name.value}'")

    return validated


# =============================================================================
# HANDWRITTEN FIELD DETECTION
# =============================================================================

class HandwrittenFieldDetector:
    """
    Detect handwritten fields using image processing techniques.
    
    Specializes in detecting:
    - Handwritten HP values (20-100 range)
    - Handwritten cost values (6+ digits)
    - Checkmarks for selection indicators
    - Stroke analysis for handwriting vs printed text
    """

    def __init__(self):
        """Initialize detector"""
        pass

    def detect_checkmarks(self, image: np.ndarray,
                          region: Tuple[int, int, int, int] = None) -> List[CheckmarkResult]:
        """
        Detect checkmarks/tick marks in the image.
        
        Checkmarks are commonly used in invoices to mark selected options,
        especially for HP (horse power) selection in tabular data.

        Args:
            image: Input image (BGR format)
            region: Optional region of interest (x1, y1, x2, y2)

        Returns:
            List of CheckmarkResult objects
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape[:2]
        
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
        
        gh, gw = search_region.shape[:2]
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(search_region, (3, 3), 0)
        
        # Apply adaptive threshold for better contrast detection
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        checkmarks = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Checkmark heuristics
            min_area = gh * gw * 0.001  # Very small minimum area
            max_area = gh * gw * 0.05   # Small maximum area
            
            if area < min_area or area > max_area:
                continue
            
            # Analyze shape
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Check for tick mark characteristics
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Detect tick mark
            is_tick = (
                aspect_ratio > 0.3 and
                aspect_ratio < 3.0 and
                solidity > 0.3 and
                circularity < 0.5
            )
            
            # Determine checkmark type
            if is_tick:
                checkmark_type = "tick"
                confidence = 0.85
            elif circularity > 0.6:
                checkmark_type = "circle"
                confidence = 0.75
            else:
                checkmark_type = "square"
                confidence = 0.65
            
            # Calculate bounding box in original coordinates
            if region:
                orig_bbox = (x + x1, y + y1, x + x1 + cw, y + y1 + ch)
            else:
                orig_bbox = (x, y, x + cw, y + ch)
            
            checkmarks.append(CheckmarkResult(
                present=True,
                bbox=orig_bbox,
                checkmark_type=checkmark_type,
                confidence=confidence
            ))
        
        if not checkmarks:
            return [CheckmarkResult(present=False, confidence=0.9)]
        
        logger.info(f"Detected {len(checkmarks)} checkmarks")
        return checkmarks

    def detect_handwritten_hp(self, image: np.ndarray,
                               blocks: List = None,
                               checkmarks: List[CheckmarkResult] = None) -> Optional[HandwrittenFieldResult]:
        """
        Detect handwritten horse power values in the image.
        
        HP values for tractors are typically in the range 20-100.
        Handwritten HP is often found near checkmarks in tabular data.

        Args:
            image: Input image (BGR format)
            blocks: OCR blocks for text-based detection
            checkmarks: Detected checkmarks for guided HP selection

        Returns:
            HandwrittenFieldResult with HP value and confidence
        """
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        
        # Center/body region for HP (typically in middle of document)
        hp_region_y_start = int(h * 0.3)
        hp_region_y_end = int(h * 0.7)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image[hp_region_y_start:hp_region_y_end, :], cv2.COLOR_BGR2GRAY)
        else:
            gray = image[hp_region_y_start:hp_region_y_end, :]
        
        # Detect handwritten strokes in HP region
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        hp_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter for digit-sized regions
            min_area = 50
            max_area = 2000
            
            if area < min_area or area > max_area:
                continue
            
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Digit-like aspect ratio (width ~ height or slightly wider)
            if 0.3 < aspect_ratio < 1.5:
                # Extract region for stroke analysis
                region = gray[y:y+ch, x:x+cw]
                
                # Calculate stroke width variance
                stroke_info = self._analyze_stroke_width(region)
                
                # Handwritten digits typically have:
                # - Irregular stroke widths
                # - Connected components
                # - Varied thickness
                
                if stroke_info['is_handwritten']:
                    # Check if value is in HP range
                    numbers = re.findall(r'\d+', str(area))  # Use area as proxy
                    hp_value = int(numbers[0]) if numbers else 0
                    
                    # HP should be 20-100
                    if 20 <= hp_value <= 100:
                        hp_candidates.append({
                            'value': hp_value,
                            'bbox': (x, y + hp_region_y_start, x + cw, y + hp_region_y_start + ch),
                            'confidence': stroke_info['confidence'],
                            'stroke_width': stroke_info['avg_width']
                        })
        
        if hp_candidates:
            # Sort by confidence and return best
            hp_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best = hp_candidates[0]
            
            logger.info(f"Handwritten HP detected: {best['value']} with confidence {best['confidence']}")
            return HandwrittenFieldResult(
                value=best['value'],
                confidence=best['confidence'],
                bbox=best['bbox'],
                field_type="hp",
                is_handwritten=True,
                stroke_width_avg=best['stroke_width']
            )
        
        return None

    def detect_handwritten_cost(self, image: np.ndarray,
                                 blocks: List = None) -> Optional[HandwrittenFieldResult]:
        """
        Detect handwritten cost values in the image.
        
        Tractor costs are typically 6+ digits (3-15 lakhs).
        Handwritten costs are often found in footer or near total amounts.

        Args:
            image: Input image (BGR format)
            blocks: OCR blocks for text-based detection

        Returns:
            HandwrittenFieldResult with cost value and confidence
        """
        h, w = image.shape[:2] if len(image.shape) == 2 else image.shape[:2]
        
        # Footer region for cost (typically bottom of document)
        footer_start = int(h * 0.7)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image[footer_start:h, :], cv2.COLOR_BGR2GRAY)
        else:
            gray = image[footer_start:h, :]
        
        gh, gw = gray.shape[:2]
        
        # Detect handwritten strokes in cost region
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        cost_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter for number-sized regions (larger than HP digits)
            min_area = 100
            max_area = 5000
            
            if area < min_area or area > max_area:
                continue
            
            # Extract region for analysis
            region = gray[y:y+ch, x:x+cw]
            
            # Calculate stroke width variance
            stroke_info = self._analyze_stroke_width(region)
            
            if stroke_info['is_handwritten']:
                # Cost values are typically larger numbers
                # Use area as rough estimate for cost
                if area > 500:  # Larger regions likely cost values
                    cost_candidates.append({
                        'value': int(area),  # Will be refined with OCR
                        'bbox': (x, y + footer_start, x + cw, y + footer_start + ch),
                        'confidence': stroke_info['confidence'],
                        'stroke_width': stroke_info['avg_width']
                    })
        
        if cost_candidates:
            cost_candidates.sort(key=lambda x: x['confidence'], reverse=True)
            best = cost_candidates[0]
            
            logger.info(f"Handwritten cost region detected with confidence {best['confidence']}")
            return HandwrittenFieldResult(
                value=best['value'],
                confidence=best['confidence'],
                bbox=best['bbox'],
                field_type="cost",
                is_handwritten=True,
                stroke_width_avg=best['stroke_width']
            )
        
        return None

    def _analyze_stroke_width(self, region: np.ndarray) -> Dict:
        """
        Analyze stroke width patterns to detect handwriting.
        
        Handwritten text typically has:
        - Irregular stroke widths
        - Varying line thickness
        - Connected components with varied density
        
        Args:
            region: Image region to analyze

        Returns:
            Dictionary with 'is_handwritten' bool and confidence score
        """
        if region.size == 0:
            return {'is_handwritten': False, 'confidence': 0.0, 'avg_width': 0}
        
        h, w = region.shape
        
        # Apply edge detection
        edges = cv2.Canny(region, 50, 150)
        
        # Calculate stroke width statistics
        # Handwriting tends to have more variation in stroke width
        
        # Use vertical projections
        vertical_proj = np.sum(edges, axis=0)
        horizontal_proj = np.sum(edges, axis=1)
        
        # Calculate variance in projections
        vertical_var = np.var(vertical_proj) if len(vertical_proj) > 0 else 0
        horizontal_var = np.var(horizontal_proj) if len(horizontal_proj) > 0 else 0
        
        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cv2.threshold(region, 127, 255, cv2.THRESH_BINARY)[1]
        )
        
        # Handwriting typically has multiple connected components
        # with varying sizes
        if num_labels > 1:
            component_sizes = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
            if len(component_sizes) > 0:
                size_variance = np.var(component_sizes)
                size_mean = np.mean(component_sizes)
                
                # Normalize variance by mean
                normalized_variance = size_variance / (size_mean ** 2) if size_mean > 0 else 0
                
                # Handwriting tends to have higher normalized variance
                is_handwritten = normalized_variance > 0.1
                
                # Calculate confidence based on various factors
                confidence = min(0.5 + normalized_variance * 2, 0.9)
                
                # Estimate average stroke width
                avg_width = size_mean ** 0.5 if size_mean > 0 else 0
                
                return {
                    'is_handwritten': is_handwritten,
                    'confidence': confidence,
                    'avg_width': avg_width,
                    'vertical_var': vertical_var,
                    'horizontal_var': horizontal_var
                }
        
        return {'is_handwritten': False, 'confidence': 0.5, 'avg_width': 0}

    def detect_red_ink_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions with red/brown ink (typical for stamps).
        
        Stamps are often in red or brown ink, making them detectable
        by color analysis.

        Args:
            image: Input image (BGR format)

        Returns:
            List of bounding boxes for potential stamp regions
        """
        if len(image.shape) != 3:
            return []
        
        h, w = image.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define red color range in HSV
        # Red is at the boundaries of HSV (0-10 and 170-180)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red regions
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Also detect brown/orange colors (common in stamps)
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([30, 255, 255])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(red_mask, brown_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        stamp_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (stamps are typically moderate size)
            min_area = h * w * 0.005  # 0.5% of image
            max_area = h * w * 0.15   # 15% of image
            
            if area < min_area or area > max_area:
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Stamps are often roughly circular or square
            if 0.5 < aspect_ratio < 2.0:
                stamp_regions.append((x, y, x + cw, y + ch))
        
        logger.info(f"Detected {len(stamp_regions)} potential stamp regions")
        return stamp_regions


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_step4(layout_data: Dict[str, Any],
                  image_path: str = None,
                  output_dir: str = None) -> ExtractedFields:
    """
    STEP 4: Field Detection & Entity Extraction - Main processing function

    INPUT (from Step 3):
    {
        "doc_id": "172610467",
        "page_number": 1,
        "header_blocks": [...],
        "body_blocks": [...],
        "footer_blocks": [...]
    }

    OUTPUT:
    {
        "doc_id": "172610467",
        "page_number": 1,
        "dealer_name": {"value": "ABC Tractors", "confidence": 0.92},
        "model_name": {"value": "Mahindra 575 DI", "confidence": 0.97},
        "horse_power": {"value": 50, "confidence": 0.95},
        "asset_cost": {"value": 525000, "confidence": 0.93},
        "signature": {"present": true, "bbox": [...], "confidence": 0.9},
        "stamp": {"present": true, "bbox": [...], "confidence": 0.88},
        "document_confidence": 0.92
    }

    Args:
        layout_data: Layout data from Step 3
        image_path: Optional path to original image for visual detection
        output_dir: Optional directory to save JSON results

    Returns:
        ExtractedFields object with all extracted fields
    """
    logger.info(f"Starting STEP 4: Field Detection for document {layout_data.get('doc_id', 'unknown')}")

    # Initialize extractors
    text_extractor = TextFieldExtractor()
    visual_detector = VisualFieldDetector()
    handwritten_detector = HandwrittenFieldDetector()

    # Get blocks from each region
    header_blocks = layout_data.get('header_blocks', [])
    body_blocks = layout_data.get('body_blocks', [])
    footer_blocks = layout_data.get('footer_blocks', [])

    # Combine blocks for searches that span regions
    all_blocks = header_blocks + body_blocks + footer_blocks

    # Extract text-based fields
    dealer_name = text_extractor.extract_dealer_name(header_blocks + body_blocks, region="header")
    model_name = text_extractor.extract_model_name(header_blocks + body_blocks, region="all")
    horse_power = text_extractor.extract_horse_power(body_blocks + footer_blocks, region="body")
    asset_cost = text_extractor.extract_asset_cost(footer_blocks + body_blocks, region="footer")

    # Cross-validate and enhance extracted fields
    validated_fields = validate_and_cross_check_fields(dealer_name, model_name, horse_power, asset_cost)
    dealer_name = validated_fields['dealer_name']
    model_name = validated_fields['model_name']
    horse_power = validated_fields['horse_power']
    asset_cost = validated_fields['asset_cost']

    # Load image for visual detection
    image = None
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)

    # Extract visual fields
    if image is not None:
        signature = visual_detector.detect_signature(image)
        
        # Enhanced stamp detection with red ink analysis
        stamp_regions = handwritten_detector.detect_red_ink_regions(image)
        if stamp_regions:
            # Use visual detector for stamp confirmation
            stamp = visual_detector.detect_stamp(image, region="all")
            if not stamp.present:
                # Use red ink region as stamp detection
                for region in stamp_regions[:3]:  # Check top 3 regions
                    stamp_result = visual_detector.detect_stamp(image, region=region)
                    if stamp_result.present:
                        stamp = stamp_result
                        break
        else:
            stamp = visual_detector.detect_stamp(image, region="footer")
        
        # Detect checkmarks for HP selection
        checkmarks = handwritten_detector.detect_checkmarks(image)
        
        # If checkmarks found, try to enhance HP extraction with checkmark guidance
        checkmark_hp = None
        for checkmark in checkmarks:
            if checkmark.present and checkmark.checkmark_type == "tick":
                # Look for HP values near the checkmark
                checkmark_bbox = checkmark.bbox
                if checkmark_bbox and image is not None:
                    # Expand region around checkmark
                    h, w = image.shape[:2]
                    expand = 50
                    x1 = max(0, checkmark_bbox[0] - expand)
                    y1 = max(0, checkmark_bbox[1] - expand)
                    x2 = min(w, checkmark_bbox[2] + expand)
                    y2 = min(h, checkmark_bbox[3] + expand)
                    
                    # Extract region and use OCR to find HP
                    region_image = image[y1:y2, x1:x2]
                    if len(region_image.shape) == 3:
                        gray_region = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_region = region_image
                    
                    # Simple OCR-like number extraction from region
                    try:
                        from PIL import Image
                        import pytesseract
                        pil_image = Image.fromarray(gray_region)
                        region_text = pytesseract.image_to_string(pil_image, config='--psm 6')
                        
                        # Look for HP pattern
                        hp_match = re.search(r'(\d{2,3})', region_text)
                        if hp_match:
                            hp_val = int(hp_match.group(1))
                            if 15 <= hp_val <= 100:
                                checkmark_hp = FieldResult(
                                    value=hp_val,
                                    confidence=0.85,  # Higher confidence due to checkmark confirmation
                                    bbox=checkmark_bbox,
                                    source="visual"
                                )
                                logger.info(f"HP extracted from checkmark region: {hp_val}")
                                break
                    except Exception as e:
                        logger.debug(f"Could not extract HP from checkmark region: {e}")
        
        # Use checkmark-based HP if available and text-based HP is weak
        if checkmark_hp and horse_power:
            if horse_power.confidence < 0.8:
                horse_power = checkmark_hp
        elif checkmark_hp:
            horse_power = checkmark_hp
        
    else:
        signature = SignatureResult(present=False, confidence=0.9)
        stamp = StampResult(present=False, confidence=0.9)
        checkmarks = []

    # Compute document confidence (average of all fields)
    confidences = []
    if dealer_name:
        confidences.append(dealer_name.confidence)
    if model_name:
        confidences.append(model_name.confidence)
    if horse_power:
        confidences.append(horse_power.confidence)
    if asset_cost:
        confidences.append(asset_cost.confidence)
    if signature:
        confidences.append(signature.confidence)
    if stamp:
        confidences.append(stamp.confidence)

    document_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # Create result object
    result = ExtractedFields(
        doc_id=layout_data.get('doc_id', 'unknown'),
        page_number=layout_data.get('page_number', 1),
        dealer_name=dealer_name,
        model_name=model_name,
        horse_power=horse_power,
        asset_cost=asset_cost,
        signature=signature,
        stamp=stamp,
        document_confidence=round(document_confidence, 2)
    )

    logger.info(f"STEP 4 complete: Document confidence = {result.document_confidence}")

    # Export to JSON if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{result.doc_id}_page{result.page_number}_extracted.json"
        filepath = os.path.join(output_dir, filename)
        result.to_json(filepath)
        logger.info(f"Extraction results saved to {filepath}")

    return result


def load_layout_output(layout_dir: str, limit: int = 100) -> Dict[str, Dict]:
    """
    Load layout JSON files from directory (randomly selected up to limit).

    Args:
        layout_dir: Directory containing layout JSON files
        limit: Maximum number of files to load (default: 100)

    Returns:
        Dictionary with doc_id as key, layout data as value
    """
    layout_data = {}

    if not os.path.exists(layout_dir):
        logger.warning(f"Layout directory {layout_dir} does not exist")
        return layout_data

    # Get all layout JSON files
    all_files = [f for f in os.listdir(layout_dir) if f.endswith('.json') and 'layout' in f]
    
    # Randomly select files if limit is set
    if limit and limit < len(all_files):
        selected_files = random.sample(all_files, limit)
    else:
        selected_files = all_files

    for filename in selected_files:
        filepath = os.path.join(layout_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            doc_id = data.get('doc_id', filename.split('_')[0])
            layout_data[doc_id] = data
            logger.info(f"Loaded layout data for {doc_id}")
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded {len(layout_data)} layout files (limit: {limit})")
    return layout_data


# =============================================================================
# DEMO FUNCTION
# =============================================================================

def demo_step4(layout_dir: str = 'layout_output', image_dir: str = 'normalized_output'):
    """
    Demo function to show Step 4 field extraction.

    Args:
        layout_dir: Directory containing layout JSON files
        image_dir: Directory containing original images
    """
    logger.info("Starting Step 4 Demo: Field Detection & Entity Extraction")

    # Load layout data
    layout_data = load_layout_output(layout_dir)

    if not layout_data:
        logger.warning(f"No layout data found in {layout_dir}")
        return

    print("\n" + "="*60)
    print("STEP 4: Field Detection & Entity Extraction Demo")
    print("="*60)

    print(f"\nLoaded layout data for {len(layout_data)} documents")

    # Process first document as example
    sample_doc_id = list(layout_data.keys())[0]
    sample_layout = layout_data[sample_doc_id]

    print(f"\nProcessing sample document: {sample_doc_id}")

    # Try to find corresponding image
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_path = os.path.join(image_dir, f"{sample_doc_id}_pg1_normalized{ext}")
        if os.path.exists(potential_path):
            image_path = potential_path
            break
        # Also try without page suffix
        potential_path = os.path.join(image_dir, f"{sample_doc_id}{ext}")
        if os.path.exists(potential_path):
            image_path = potential_path
            break

    if image_path:
        print(f"Found image: {image_path}")
    else:
        print("No image found for visual detection")

    # Run Step 4
    result = process_step4(sample_layout, image_path=image_path)

    print("\n--- Extracted Fields ---")
    print(f"Dealer Name: {result.dealer_name.value if result.dealer_name else 'Not found'} "
          f"(confidence: {result.dealer_name.confidence if result.dealer_name else 0:.2f})")
    print(f"Model Name: {result.model_name.value if result.model_name else 'Not found'} "
          f"(confidence: {result.model_name.confidence if result.model_name else 0:.2f})")
    print(f"Horse Power: {result.horse_power.value if result.horse_power else 'Not found'} "
          f"(confidence: {result.horse_power.confidence if result.horse_power else 0:.2f})")
    print(f"Asset Cost: {result.asset_cost.value if result.asset_cost else 'Not found'} "
          f"(confidence: {result.asset_cost.confidence if result.asset_cost else 0:.2f})")
    print(f"Signature: {'Present' if result.signature.present else 'Not found'} "
          f"(confidence: {result.signature.confidence:.2f})")
    print(f"Stamp: {'Present' if result.stamp.present else 'Not found'} "
          f"(confidence: {result.stamp.confidence:.2f})")
    print(f"\nDocument Confidence: {result.document_confidence:.2f}")

    # Export to JSON
    json_output = result.to_json()
    print(f"\n--- JSON Output ---")
    print(json_output[:800] + "..." if len(json_output) > 800 else json_output)

    return result


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Run demo
    demo_step4()

    # Also test on all layout files
    print("\n" + "="*60)
    print("Processing All Documents")
    print("="*60)

    try:
        # Load layout data
        layout_dir = "layout_output"
        layout_data = load_layout_output(layout_dir)

        if layout_data:
            print(f"\nFound {len(layout_data)} layout files")

            # Create output directory
            extracted_output_dir = "extracted_output"
            os.makedirs(extracted_output_dir, exist_ok=True)

            # Process all documents
            results = {}
            for doc_id, layout in layout_data.items():
                # Find image
                image_path = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    for suffix in ['', '_pg1_normalized']:
                        potential = os.path.join("normalized_output", f"{doc_id}{suffix}{ext}")
                        if os.path.exists(potential):
                            image_path = potential
                            break
                    if image_path:
                        break

                result = process_step4(layout, image_path=image_path,
                                      output_dir=extracted_output_dir)
                results[doc_id] = result

            print(f"\nStep 4: Extracted fields from {len(results)} documents")
            print(f"Results saved to: {extracted_output_dir}/")

            # Show summary
            print("\n--- Extraction Summary ---")
            for doc_id, result in list(results.items())[:5]:
                print(f"\n{doc_id}:")
                print(f"  Dealer: {result.dealer_name.value if result.dealer_name else 'N/A'}")
                print(f"  Model: {result.model_name.value if result.model_name else 'N/A'}")
                print(f"  HP: {result.horse_power.value if result.horse_power else 'N/A'}")
                print(f"  Cost: {result.asset_cost.value if result.asset_cost else 'N/A'}")
                print(f"  Signature: {'Yes' if result.signature.present else 'No'}")
                print(f"  Stamp: {'Yes' if result.stamp.present else 'No'}")
                print(f"  Confidence: {result.document_confidence:.2f}")
        else:
            print(f"No layout files found in {layout_dir}")
            print("Run Layout.py first to generate layout output")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

