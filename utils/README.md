# Utils - Invoice Processing Utility Modules

Location: All modules are in `/root/invoicement/utils/`

## Pipeline Modules

| File | Step | Purpose |
|------|------|---------|
| `10_Normalize.py` | 1 | Image preprocessing |
| `20_OCR.py` | 2 | Text extraction with OCR |
| `30_Layout.py` | 3 | Spatial layout analysis |
| `40_Extract.py` | 4 | Field extraction |
| `50_Validation.py` | 5 | Validation |
| `60_Output.py` | 6 | Output formatting |

## Utility Modules

| Module | Purpose |
|--------|---------|
| `handwriting.py` | Stroke width analysis for handwriting detection |
| `symbols.py` | Checkmark and symbol detection |
| `color_analysis.py` | HSV color analysis for stamp detection |

## Usage

```python
# Import pipeline steps
from utils import process_step1, process_step2, process_step3, process_step4

# Import utility functions
from utils import detect_checkmarks, analyze_stroke_width, detect_red_ink_regions
```

