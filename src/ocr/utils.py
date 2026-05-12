import re
import numpy as np
from difflib import get_close_matches

THAI_PROVINCES: list[str] = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี",
]

def normalize_text(text: str) -> str:
    """Keep only Thai characters and digits."""
    return re.sub(r'[^\u0e00-\u0e7f0-9]', '', text)


def clean_top(text: str) -> str:
    """
    Enforce Thai license-plate top-line structure.

    Valid formats:
      - Pure digit plate (old-style): up to 7 digits       e.g. 737378
      - Thai prefix: 0–1 digit + 2 Thai chars + 2–4 digits e.g. กม8300, 9กณ428
    """
    thai_chars = re.findall(r'[ก-ฮ]', text)
    digits     = re.findall(r'\d', text)

    if not thai_chars:
        return ''.join(digits[:7])

    first_thai_pos = next(
        (i for i, c in enumerate(text) if '\u0e00' <= c <= '\u0e7f'), len(text)
    )
    leading_digits = re.findall(r'\d', text[:first_thai_pos])
    last_thai_pos  = max(i for i, c in enumerate(text) if '\u0e00' <= c <= '\u0e7f')
    trail_digits   = re.findall(r'\d', text[last_thai_pos + 1:])

    lead  = ''.join(leading_digits[:1])
    thai  = ''.join(thai_chars[:2])
    trail = ''.join(trail_digits[:4])
    return lead + thai + trail


def snap_province(text: str, cutoff: float = 0.6) -> str:
    """Fuzzy-snap OCR text to the nearest valid Thai province name."""
    if text in THAI_PROVINCES:
        return text

    matches = get_close_matches(text, THAI_PROVINCES, n=1, cutoff=cutoff)
    if matches:
        return matches[0]

    prefix = text[:3] if len(text) >= 3 else text
    prefix_matches = [p for p in THAI_PROVINCES if p.startswith(prefix)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    return text


def clean_bottom(text: str) -> str:
    """Strip non-Thai characters, then snap to the nearest province name."""
    thai_only = re.sub(r'[^\u0e00-\u0e7f]', '', text)
    if not thai_only:
        return text
    return snap_province(thai_only)