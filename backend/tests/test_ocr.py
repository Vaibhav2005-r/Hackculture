import cv2
import numpy as np
import pytest

from app.ocr.metrics import cer, levenshtein, wer
from app.ocr.pipeline import run_page_ocr
from app.ocr.postprocess import normalize_text
from app.ocr.preprocess import estimate_skew, preprocess, rotate, sauvola_binarize
from app.ocr.segment import segment_lines
from app.services.synthetic import make_page


def test_levenshtein_and_rates():
    assert levenshtein("kitten", "sitting") == 3
    assert cer("abc", "abc") == 0
    assert cer("abd", "abc") == pytest.approx(1 / 3)
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)
    assert cer("", "") == 0


def test_normalize_dandas_and_avagraha():
    assert normalize_text("रामः|", "devanagari") == "रामः ।"
    assert normalize_text("सङ्गोsस्त्व", "devanagari") == "सङ्गोऽस्त्व"
    assert normalize_text("abc  def", "latin") == "abc def"


@pytest.mark.parametrize("angle", [-4.0, -1.5, 2.0, 3.5])
def test_skew_is_recovered(angle):
    page = make_page("devanagari", seed=3, difficulty=0.0).image
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    ink = sauvola_binarize(gray)
    rotated = rotate(ink, angle, 0)
    assert estimate_skew(rotated) == pytest.approx(-angle, abs=0.6)


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_segmentation_finds_every_line(seed):
    page = make_page("devanagari", seed=seed)
    boxes = segment_lines(preprocess(page.image).binary)
    assert len(boxes) == len(page.lines)


def test_frame_and_blank_page():
    blank = np.full((400, 600, 3), 230, np.uint8)
    cv2.rectangle(blank, (5, 5), (594, 394), (20, 20, 20), 3)  # a ruled border only
    assert segment_lines(preprocess(blank).binary) == []


def test_stock_baseline_reads_text_after_rec_only_calls():
    """Regression: the stock detector must still run after the engine was used in rec-only mode."""
    from app.ocr.engines import get_engine

    eng = get_engine("ppocr")
    page = make_page("devanagari", seed=1000)
    eng.recognize([page.image[:60, :400]], "devanagari")  # rec-only call first
    text = eng.read_page_stock(page.image, "devanagari")
    assert cer(text, " ".join(page.lines)) < 0.9


def test_pipeline_beats_raw_baseline():
    total_full = total_raw = 0.0
    for seed in (21, 22, 23):
        page = make_page("devanagari", seed=seed)
        truth = " ".join(page.lines)
        full = run_page_ocr(page.image, "devanagari", "ppocr")
        raw = run_page_ocr(page.image, "devanagari", "ppocr", use_preprocessing=False)
        total_full += cer(" ".join(ln.text for ln in full.lines), truth)
        total_raw += cer(" ".join(ln.text for ln in raw.lines), truth)
    assert total_full / 3 < 0.15
    assert total_full < total_raw


def test_rover_keeps_pivot_length_and_handles_insertions():
    from app.ocr.engines.ensemble import rover_vote

    # The strongest candidate is the pivot; an extra character in another candidate must not
    # shift or truncate the result, and two weaker votes outweigh one wrong pivot character.
    text, conf = rover_vote([("abXd", 0.9), ("abcd", 0.8), ("abcdZZ", 0.8)], [0.4, 0.3, 0.3])
    assert text == "abcd"
    assert 0.0 < conf <= 1.0
    assert rover_vote([("", 0.0), ("abc", 0.7)]) == ("abc", 0.7)
    assert rover_vote([]) == ("", 0.0)
