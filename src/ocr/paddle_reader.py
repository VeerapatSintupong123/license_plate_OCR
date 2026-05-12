import numpy as np
from paddleocr import PaddleOCR
from utils import normalize_text, clean_top, clean_bottom

class PaddleReader:
    """
    OCR engine for Thai license plates using PaddleOCR.

    Splits detected text regions into *top line* (plate number) and
    *bottom line* (province), then applies post-processing to return
    clean, structured results.

    Primary usage in pipeline (called by main.py)::

        ocr_engine = PaddleReader()
        plate_text, province = ocr_engine.extract_text(aligned_plate)
    """

    def __init__(
        self,
        lang: str = "th",
        conf_threshold: float = 0.5,
    ):
        """
        Args:
            lang:            PaddleOCR language code (default: 'th' for Thai).
            conf_threshold:  Minimum OCR confidence to accept a text region.
        """
        self.conf_threshold = conf_threshold
        self._ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    def extract_text(self, image: np.ndarray) -> tuple[str, str]:
        """
        Run OCR on a pre-warped/cropped license-plate image and return the
        plate number and province as separate strings.

        Called by main.py::
            plate_text, province = ocr_engine.extract_text(aligned_plate)

        Args:
            image: BGR numpy array of the (perspective-corrected) plate crop.

        Returns:
            (plate_text, province) where:
                plate_text – cleaned top-line text, e.g. "กม8300"
                province   – snapped Thai province name, e.g. "กรุงเทพมหานคร"
                             (empty string if not detected)
        """
        raw         = self._ocr.predict(image)
        detections  = self._parse_paddle_result(raw)
        top, bottom = self._split_lines(detections, image.shape[0])

        top_raw    = ''.join(d['text'] for d in top)
        bottom_raw = ''.join(d['text'] for d in bottom)

        plate_text = clean_top(top_raw)    if top_raw    else ""
        province   = clean_bottom(bottom_raw) if bottom_raw else ""

        return plate_text, province

    def _parse_paddle_result(self, result) -> list[dict]:
        """
        Convert raw PaddleOCR output to a flat list of detection dicts.

        Each dict contains:
            text  – raw recognized string
            score – confidence float
            bbox  – [x1, y1, x2, y2] axis-aligned bounding box (int pixels)
        """
        if not result or not result[0]:
            return []

        res    = result[0]   # first (and only) image
        output = []

        for text, score, poly in zip(
            res["rec_texts"], res["rec_scores"], res["rec_polys"]
        ):
            if float(score) < self.conf_threshold:
                continue

            clean = normalize_text(text)
            if not clean:
                continue

            pts = np.array(poly)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())

            output.append({
                "text":  clean,
                "score": float(score),
                "bbox":  [x1, y1, x2, y2],
            })

        return output

    def _split_lines(
        self,
        detections: list[dict],
        image_height: int,
    ) -> tuple[list[dict], list[dict]]:
        """
        Divide detections into top (plate number) and bottom (province) lines.

        Strategy: the plate number sits in the upper half of the crop and the
        province in the lower half.  Detections are then sorted left-to-right
        within each line so concatenation preserves reading order.

        Args:
            detections:   Output of _parse_paddle_result.
            image_height: Height of the plate crop in pixels.

        Returns:
            (top_detections, bottom_detections)
        """
        midpoint = image_height / 2

        top    = []
        bottom = []

        for det in detections:
            y_center = (det["bbox"][1] + det["bbox"][3]) / 2
            if y_center <= midpoint:
                top.append(det)
            else:
                bottom.append(det)

        top.sort(key=lambda d: d["bbox"][0])
        bottom.sort(key=lambda d: d["bbox"][0])

        return top, bottom