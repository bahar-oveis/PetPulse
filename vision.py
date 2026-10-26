"""
PetPulse Vision: local image processing and OCR for food labels.

Pipeline (all on this computer):
1. Validate the upload: real JPEG/PNG data (not just the file name), size and
   pixel limits, not corrupted.
2. Fix orientation from EXIF and shrink very large photos (keeping the aspect ratio).
3. Estimate sharpness with OpenCV's Laplacian variance (a rough blur check).
4. Make an OCR-friendly copy: grayscale, upscaled if small, contrast boosted with CLAHE.
5. Read the text with Tesseract (pytesseract), with an average word confidence.

A normal color copy is kept for Claude, because the OCR copy loses useful detail.
Images are processed in memory and never saved.
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PIXELS = 40_000_000              # e.g. 8000 x 5000
MIN_SIDE = 200                       # smaller images can't show readable ingredient text
WORKING_MAX_SIDE = 2400              # enough for small print, without slowing OCR down
CLAUDE_MAX_SIDE = 1568               # larger images are downscaled by Claude anyway
BLUR_THRESHOLD = 100.0               # Laplacian variance below this looks blurry
MIN_OCR_CONFIDENCE = 60              # average Tesseract word confidence (0-100)
MIN_OCR_WORDS = 5
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png"}


class VisionError(Exception):
    """An image problem the sitter should see as a friendly message."""


@dataclass
class PreparedImage:
    color: Image.Image       # orientation-corrected, resized RGB copy
    ocr_image: np.ndarray    # grayscale, contrast-enhanced copy for Tesseract
    blur_score: float
    is_blurry: bool


@dataclass
class OcrResult:
    text: str
    confidence: float  # average word confidence, 0-100
    word_count: int

    @property
    def readable(self):
        return self.word_count >= MIN_OCR_WORDS and self.confidence >= MIN_OCR_CONFIDENCE


def load_image(data):
    """Check that data is a supported, undamaged image and return it as an upright RGB image."""
    if not data:
        raise VisionError("Please upload or take a photo of the food label first.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise VisionError(f"The image is too large. Please use one under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    try:
        with Image.open(io.BytesIO(data)) as probe:  # reads the header: the real format, not the file name
            image_format, (width, height) = probe.format, probe.size
            probe.verify()                           # checks the file structure
    except Image.DecompressionBombError:
        raise VisionError("The image has too many pixels. Please use a smaller photo.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise VisionError("This file isn't a valid image, or it's damaged. Please use a JPG or PNG photo.")

    if image_format not in ALLOWED_FORMATS:
        raise VisionError("Please use a JPG or PNG image.")
    if width * height > MAX_PIXELS:
        raise VisionError("The image has too many pixels. Please use a smaller photo.")
    if min(width, height) < MIN_SIDE:
        raise VisionError("The image is too small to read the label. Please use a larger, closer photo.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # decodes all pixels, which catches truncated files
        image = ImageOps.exif_transpose(image)  # phone photos store rotation in EXIF
    except (OSError, SyntaxError, ValueError):
        raise VisionError("This image is damaged and can't be read. Please take or upload it again.")

    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, "white")  # transparent PNGs: put them on white
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def resize_to_fit(image, max_side):
    """Shrink so the longest side is at most max_side, keeping the aspect ratio. Never enlarges."""
    if max(image.size) <= max_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _gray(image):
    import cv2

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)


def blur_score(image):
    """
    Sharpness estimate: variance of the Laplacian (edge strength) at a fixed width.
    Low values suggest a blurry photo. It's only a heuristic: a sharp photo can still be hard to read.
    """
    import cv2

    gray = _gray(image)
    scale = 1000 / gray.shape[1]
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def enhance_for_ocr(image):
    """Grayscale, enlarge small images, and even out contrast (CLAHE) for Tesseract."""
    import cv2

    gray = _gray(image)
    if max(gray.shape) < 1500:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # small print reads better larger
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def prepare_label_image(data):
    """Run the local preprocessing pipeline on uploaded image bytes."""
    color = resize_to_fit(load_image(data), WORKING_MAX_SIDE)
    score = blur_score(color)
    return PreparedImage(color=color, ocr_image=enhance_for_ocr(color), blur_score=score,
                         is_blurry=score < BLUR_THRESHOLD)


def extract_ingredients_text(image):
    """Read text from an image with local Tesseract OCR (English). Returns an OcrResult."""
    import pytesseract

    try:
        data = pytesseract.image_to_data(image, lang="eng", output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        raise VisionError("Text recognition needs Tesseract, which isn't installed. Run: brew install tesseract")
    except (pytesseract.TesseractError, RuntimeError, OSError):
        raise VisionError("The label text couldn't be read. Please try a clearer photo.")

    lines, confidences = {}, []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        if not word:
            continue
        line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(line_id, []).append(word)
        confidence = float(data["conf"][i])
        if confidence >= 0:
            confidences.append(confidence)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    average = round(sum(confidences) / len(confidences), 1) if confidences else 0.0
    return OcrResult(text=text, confidence=average, word_count=len(confidences))


def image_for_claude(image):
    """A JPEG copy (longest side at most 1568 px) to send to Claude, plus its media type."""
    buffer = io.BytesIO()
    resize_to_fit(image, CLAUDE_MAX_SIDE).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), "image/jpeg"
