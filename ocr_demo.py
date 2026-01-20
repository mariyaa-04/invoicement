from pathlib import Path
from utils.document_loader import load_document_as_images
from utils.schemas import OCRResult
from utils.layout_analysis import split_sections, detect_tables, associate_key_value
from utils.candidate_extraction import find_candidates, extract_hp_candidates, extract_cost_candidates, KEYWORDS
from utils.validation import validate_hp, validate_cost, validate_presence, validate_confidence
from utils.field_selection import select_best_candidates
from utils.visualization import plot_confidence_distribution, plot_field_candidates

SAMPLE_DIR = Path("data/sample_invoices")


def mock_paddleocr(_image):
    return [
        {"text": "Dealer ABC Tractors", "box": [[50, 50], [260, 50], [260, 75], [50, 75]], "score": 0.96, "page_num": 1},
        {"text": "HP 50", "box": [[50, 120], [120, 120], [120, 145], [50, 145]], "score": 0.98, "page_num": 1},
        {"text": "Cost 150000", "box": [[50, 170], [200, 170], [200, 195], [50, 195]], "score": 0.97, "page_num": 1},
        {"text": "Model X1", "box": [[50, 220], [150, 220], [150, 245], [50, 245]], "score": 0.95, "page_num": 1}
    ]


def standardize_ocr_output(raw_ocr):
    results = []
    for item in raw_ocr:
        results.append(
            OCRResult(text=item["text"], bbox=item["box"], confidence=item["score"], page=item["page_num"])
        )
    return results


def process_invoice(file_path):
    images = load_document_as_images(str(file_path))
    raw_ocr = []
    for img in images:
        raw_ocr.extend(mock_paddleocr(img))

    ocr_results = standardize_ocr_output(raw_ocr)
    ocr_results = validate_confidence(ocr_results, min_confidence=0.5)

    sections = split_sections(ocr_results, page_height=1000)
    sections.tables = detect_tables(sections.body)

    candidates = {}
    for field, keywords in KEYWORDS.items():
        nearby_words = find_candidates(ocr_results, keywords)
        if field == "HP":
            candidates[field] = extract_hp_candidates(nearby_words)
        elif field == "cost":
            candidates[field] = extract_cost_candidates(nearby_words)
        else:
            candidates[field] = [w.text for w in nearby_words]

    validators = {"HP": validate_hp, "cost": validate_cost}
    final_candidates = {}
    for field, cands in candidates.items():
        scored = select_best_candidates(field, cands, ocr_results, validators)
        final_candidates[field] = [s["value"] for s in scored][:1]  # pick top candidate

    final_candidates = validate_presence(final_candidates)

    plot_confidence_distribution(ocr_results)
    plot_field_candidates(final_candidates)

    return sections, final_candidates


def main():
    files = list(SAMPLE_DIR.glob("*.*"))
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")
        sections, candidates = process_invoice(file_path)

        print("Header:", [w.text for w in sections.header])
        print("Body:", [w.text for w in sections.body])
        print("Footer:", [w.text for w in sections.footer])

        print("Tables:")
        for table in sections.tables:
            print(associate_key_value(table))

        print("Final Candidates:", candidates)
        print("-" * 50)


if __name__ == "__main__":
    main()
