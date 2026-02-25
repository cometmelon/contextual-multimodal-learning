# ============================================================
# agent/validator.py — Tiered Fusion Guardrail
# ============================================================
# Tier 1: SigLIP cosine similarity (deterministic math check)
# Tier 2: Dynamic thresholds based on visual classification
# Tier 3: LLM-Judge for abstract content gray zone
# ============================================================

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from config import key_rotator, MODEL_FLASH

# ── Lazy-loaded SigLIP Model ─────────────────────────────────
_siglip_processor = None
_siglip_model = None


def _load_siglip():
    """Lazy-load SigLIP model (only on first validation call)."""
    global _siglip_processor, _siglip_model
    if _siglip_model is None:
        print("📦 Loading SigLIP model (first-time only)...")
        _siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        _siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        _siglip_model.eval()
        print("✅ SigLIP loaded.")


def siglip_similarity(image: Image.Image, text: str) -> float:
    """
    Compute cosine similarity between an image and text using SigLIP.
    
    Args:
        image: PIL Image (the cropped snippet)
        text: Text to compare against (the short caption from the LLM)
    
    Returns:
        Cosine similarity score between 0.0 and 1.0
    """
    _load_siglip()

    inputs = _siglip_processor(
        text=[text],
        images=image,
        padding="max_length",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = _siglip_model(**inputs)

    # Normalize embeddings
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)

    # Cosine similarity
    similarity = torch.nn.functional.cosine_similarity(image_embeds, text_embeds).item()

    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def get_dynamic_thresholds(visual_label: str) -> tuple[float, float]:
    """
    Return (upper_bound, lower_bound) based on the visual classification.
    
    Photographic content: high thresholds (SigLIP is reliable)
    Abstract content (code, diagrams): low thresholds (defer to LLM-Judge)
    
    Args:
        visual_label: The classification label from Node 1
    
    Returns:
        (upper_bound, lower_bound) — above upper = pass, below lower = fail
    """
    label_lower = visual_label.lower()

    # Abstract content: code, diagrams, equations, UI layouts
    abstract_keywords = [
        "code", "script", "function", "class", "variable", "ide", "editor",
        "terminal", "console", "diagram", "uml", "flowchart", "schema",
        "equation", "formula", "math", "graph", "chart", "table",
        "spreadsheet", "ui", "interface", "layout", "wireframe",
        "whiteboard", "slide", "presentation", "text", "document",
    ]

    is_abstract = any(kw in label_lower for kw in abstract_keywords)

    if is_abstract:
        # Lean heavily on LLM-Judge for abstract content
        return (0.50, 0.20)
    else:
        # Trust SigLIP more for photographic/natural content
        return (0.75, 0.40)


def llm_judge(snippet_image: Image.Image, draft_answer: str) -> bool:
    """
    Independent LLM-Judge verification for the gray zone.
    Uses a DIFFERENT model instance to avoid correlated hallucinations.
    
    Args:
        snippet_image: The cropped snippet PIL Image
        draft_answer: The synthesized answer to verify
    
    Returns:
        True if the judge agrees the answer matches the image
    """
    import io
    
    client = key_rotator.get_client()

    # Convert PIL to bytes for Gemini API
    img_buffer = io.BytesIO()
    snippet_image.save(img_buffer, format="JPEG", quality=85)
    img_bytes = img_buffer.getvalue()

    judge_prompt = f"""You are an independent visual verification judge. Your ONLY job is to determine if the following answer accurately describes what is shown in the provided image.

ANSWER TO VERIFY: "{draft_answer}"

Rules:
1. Look at the image carefully and independently.
2. Do NOT assume the answer is correct.
3. Check if the key visual elements in the image match the answer's claims.
4. Respond with ONLY "AGREE" or "DISAGREE" followed by a one-sentence explanation.
"""

    try:
        response = key_rotator.call_with_retry(
            model=MODEL_FLASH,
            contents=[
                _pil_to_genai_part_for_judge(snippet_image),
                judge_prompt,
            ],
        )
        
        result_text = response.text.strip().upper()
        return result_text.startswith("AGREE")

    except Exception as e:
        print(f"⚠️ LLM-Judge error: {e}")
        # On judge failure, cautiously pass (don't block the answer)
        return True


def _pil_to_genai_part_for_judge(image: Image.Image):
    """
    For the new google-genai SDK, we can pass PIL Images directly 
    in the contents list. No need for base64 dictionary wrapping.
    """
    return image
