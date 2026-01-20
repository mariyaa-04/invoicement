from utils.schemas import OCRWord
from utils.candidate_extraction import extract_hp_candidates, extract_cost_candidates


def test_hp_extraction():
    words = [
        OCRWord("Power 50 HP", [0,0,0,0], 0.9, 0)
    ]
    hp = extract_hp_candidates(words)
    assert hp == [50]


def test_cost_extraction():
    words = [
        OCRWord("Total Amount 525000", [0,0,0,0], 0.9, 0)
    ]
    cost = extract_cost_candidates(words)
    assert cost == [525000]
