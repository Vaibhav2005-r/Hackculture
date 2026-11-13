"""Load every offline OCR model once so a container image ships with them (no first-request download)."""

import numpy as np

from app.ocr.engines import get_engine

blank = np.full((64, 256, 3), 255, np.uint8)
engine = get_engine("ppocr")
for script in sorted(engine.supported_scripts()):
    engine.recognize([blank], script)
    print("warmed", script)
