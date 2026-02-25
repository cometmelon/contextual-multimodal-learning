# ============================================================
# image_utils.py — Image Processing Utilities
# ============================================================
# Handles Base64 decode/encode and bounding box cropping.
# ============================================================

import base64
import io
from PIL import Image


def decode_b64_to_pil(b64_string: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) to a PIL Image."""
    # Strip data URI prefix if present (e.g., "data:image/webp;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_b64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Convert a PIL Image to a base64 string (without data URI prefix)."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 85) -> bytes:
    """Convert a PIL Image to raw bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    return buffer.getvalue()


def crop_image(full_frame_b64: str, bbox: list[float]) -> Image.Image:
    """
    Crop the bounding box region from a base64-encoded full frame.
    
    Args:
        full_frame_b64: Base64 encoded full frame image
        bbox: [x, y, width, height] coordinates relative to the displayed video
        
    Returns:
        Cropped PIL Image of the selected region
    """
    full_image = decode_b64_to_pil(full_frame_b64)
    
    x, y, w, h = bbox
    
    # Clamp coordinates to image bounds
    img_w, img_h = full_image.size
    x = max(0, int(x))
    y = max(0, int(y))
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    
    # Pillow uses (left, upper, right, lower) for crop box
    cropped = full_image.crop((x, y, x + w, y + h))
    
    return cropped
