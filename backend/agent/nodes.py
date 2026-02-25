# ============================================================
# agent/nodes.py — LangGraph Node Functions
# ============================================================
# Each function reads from and mutates the LeanAgentState.
# Images are fetched from Redis by reference — never stored in state.
# ============================================================

import io
import asyncio
from PIL import Image

from config import key_rotator, MODEL_FLASH, MODEL_PRO, MAX_CORRECTION_ATTEMPTS
from redis_client import get_image
from transcript import fetch_transcript, semantic_search_transcript
from agent.validator import siglip_similarity, get_dynamic_thresholds, llm_judge


def _bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _pil_to_genai_part(image: Image.Image):
    """
    For the new google-genai SDK, we can pass PIL Images directly 
    in the contents list. No need for base64 dictionary wrapping.
    """
    return image


# ── Node 1: Visual Labeling & Fast-Fail ───────────────────────
async def node_visual_label(state: dict) -> dict:
    """
    Pulls the snippet from Redis and asks Gemini Flash to generate
    a structural label for the cropped image.
    Also checks transcript availability (fast-fail if unavailable).
    """
    # Fetch snippet image from Redis
    snippet_bytes = await get_image(state["snippet_ref"])
    if not snippet_bytes:
        return {
            "visual_classification_label": "unknown visual content",
            "has_transcript": False,
        }

    snippet_image = _bytes_to_pil(snippet_bytes)

    # Ask Gemini Flash for a structural visual label
    client = key_rotator.get_client()

    label_prompt = """Look at this image carefully. Provide a concise structural description of what you see.

Rules:
1. Be specific about the TYPE of content (e.g., "Python code in a dark IDE", "network topology diagram", "photograph of a circuit board")
2. Keep it under 15 words.
3. Do NOT interpret or analyze the content, just classify what it visually IS.
4. Return ONLY the description, nothing else."""

    try:
        response = key_rotator.call_with_retry(
            model=MODEL_FLASH,
            contents=[
                _pil_to_genai_part(snippet_image),
                label_prompt,
            ],
        )
        visual_label = response.text.strip()
    except Exception as e:
        visual_label = f"visual content (classification failed: {str(e)[:50]})"

    # Fast-fail: Check transcript availability
    has_transcript, _ = fetch_transcript(state["video_id"])

    return {
        "visual_classification_label": visual_label,
        "has_transcript": has_transcript,
    }


# ── Node 2: Temporal Context & Semantic Transcript Search ─────
async def node_temporal_context(state: dict) -> dict:
    """
    Fetches the transcript and applies temporal windowing + semantic search.
    Self-contained: fetches transcript directly using video_id from state.
    """
    if not state.get("has_transcript"):
        return {
            "transcript_context": "[No transcript available for this video]",
        }

    # Fetch transcript directly (cached by youtube-transcript-api internally)
    has_transcript, raw_transcript = fetch_transcript(state["video_id"])

    if not has_transcript or not raw_transcript:
        return {
            "transcript_context": "[Transcript fetch failed on retry]",
        }

    transcript_text = semantic_search_transcript(
        transcript=raw_transcript,
        visual_label=state["visual_classification_label"],
        query=state["query"],
        timestamp=state["timestamp"],
    )

    return {
        "transcript_context": transcript_text or "[No relevant transcript found in temporal window]",
    }


# ── Node 3: External Tool Router (Web Search) ────────────────
async def node_tool_router(state: dict) -> dict:
    """
    Evaluates if the transcript context is sufficient.
    If not, uses Gemini to search/generate supplementary knowledge.
    """
    transcript_ctx = state.get("transcript_context", "")
    query = state["query"]
    visual_label = state["visual_classification_label"]

    # Heuristic: if transcript is missing or too short, we need external data
    needs_external = (
        not transcript_ctx
        or len(transcript_ctx) < 50
        or "[No transcript" in transcript_ctx
    )

    if not needs_external:
        # Quick check: ask the LLM if the transcript answers the query
        client = key_rotator.get_client()
        try:
            check_response = key_rotator.call_with_retry(
                model=MODEL_FLASH,
                contents=f"""Given this transcript context and visual description, can you answer the user's question?

Visual: {visual_label}
Transcript: {transcript_ctx[:500]}
Question: {query}

Respond ONLY "YES" or "NO".""",
            )
            if "NO" in check_response.text.upper():
                needs_external = True
        except Exception:
            pass

    if needs_external:
        # Use Gemini to provide supplementary knowledge
        try:
            tool_response = key_rotator.call_with_retry(
                model=MODEL_FLASH,
                contents=f"""The user is watching a YouTube video and highlighted something that looks like: "{visual_label}".
Their question is: "{query}"

The video transcript doesn't sufficiently explain this. Please provide a concise, factual explanation that would help answer their question.
Focus on technical accuracy. Keep it under 200 words.""",
            )
            return {"tool_data": tool_response.text.strip()}
        except Exception as e:
            return {"tool_data": f"[External search failed: {str(e)[:100]}]"}

    return {"tool_data": ""}


# ── Node 4: Multimodal Synthesis ──────────────────────────────
async def node_synthesize(state: dict) -> dict:
    """
    The Heavy-Duty synthesis node. GPT-4o/Gemini Pro receives:
    1. User query
    2. Full frame (from Redis)
    3. Cropped snippet (from Redis)
    4. Transcript context
    5. External tool data
    6. BBox coordinates for spatial awareness
    7. Previous error context (if correction loop)
    """
    # Fetch both images from Redis (Visual Flow — raw pixels, not text)
    full_frame_bytes = await get_image(state["full_frame_ref"])
    snippet_bytes = await get_image(state["snippet_ref"])

    if not full_frame_bytes or not snippet_bytes:
        return {"draft_answer": "Failed to retrieve image data from cache."}

    full_frame_image = _bytes_to_pil(full_frame_bytes)
    snippet_image = _bytes_to_pil(snippet_bytes)

    # Build the synthesis prompt
    correction_context = ""
    if state["correction_attempts"] > 0:
        correction_context = f"""
⚠️ CORRECTION ATTEMPT #{state['correction_attempts']}
Your previous answer was flagged by the validation guardrail as potentially inaccurate.
Previous validation score: {state['validation_score']:.2f}
Please RE-EXAMINE the images more carefully and provide a corrected answer.
Focus specifically on what the CROPPED IMAGE actually shows, not what you assume."""

    bbox = state["bbox_coordinates"]
    synthesis_prompt = f"""You are an expert AI assistant analyzing a YouTube video frame.

USER QUESTION: {state['query']}

VISUAL CONTEXT:
- Image 1 (FULL FRAME): The complete video frame for macro context
- Image 2 (CROPPED SNIPPET): The specific area the user highlighted at coordinates [x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]:.0f}, h={bbox[3]:.0f}]
- Visual Classification: {state['visual_classification_label']}

TRANSCRIPT CONTEXT (what the video creator was saying):
{state['transcript_context'][:1500]}

SUPPLEMENTARY DATA:
{state['tool_data'][:500] if state['tool_data'] else 'None needed'}

{correction_context}

INSTRUCTIONS:
1. Focus your answer on what the CROPPED SNIPPET shows
2. Use the FULL FRAME for surrounding context (what else is on screen)
3. Use the TRANSCRIPT to understand what the creator was explaining at this moment
4. Be specific, technical, and accurate
5. If you're uncertain about any detail, say so explicitly
6. Keep the answer concise but comprehensive (under 250 words)"""

    client = key_rotator.get_client()

    try:
        response = key_rotator.call_with_retry(
            model=MODEL_PRO,
            contents=[
                _pil_to_genai_part(full_frame_image),
                _pil_to_genai_part(snippet_image),
                synthesis_prompt,
            ],
        )
        draft = response.text.strip()
    except Exception as e:
        draft = f"Synthesis failed: {str(e)[:200]}"

    return {
        "draft_answer": draft,
        "correction_attempts": state["correction_attempts"] + 1,
    }


# ── Node 5: Tiered Fusion Guardrail ──────────────────────────
async def node_fusion_validator(state: dict) -> dict:
    """
    Validates the draft answer against the actual image using:
    Tier 1: SigLIP cosine similarity (math check)
    Tier 2: Dynamic thresholds based on content type
    Tier 3: LLM-Judge for gray zone verification
    """
    snippet_bytes = await get_image(state["snippet_ref"])
    if not snippet_bytes:
        # Can't validate without image — pass through
        return {"validation_score": 0.5}

    snippet_image = _bytes_to_pil(snippet_bytes)

    # Extract a short description from the draft answer for SigLIP comparison
    client = key_rotator.get_client()
    try:
        caption_response = key_rotator.call_with_retry(
            model=MODEL_FLASH,
            contents=f"""Extract ONLY a 3-5 word literal visual description of the main object from this answer. 
No analysis, just what it physically looks like.

Answer: {state['draft_answer'][:300]}

Example outputs: "Python code in dark IDE", "star network topology diagram", "red circuit board"
Respond with ONLY the description:""",
        )
        short_caption = caption_response.text.strip()
    except Exception:
        short_caption = state["visual_classification_label"]

    # Tier 1: SigLIP Math Check
    similarity = siglip_similarity(snippet_image, short_caption)

    # Tier 2: Dynamic Thresholds
    upper, lower = get_dynamic_thresholds(state["visual_classification_label"])

    if similarity >= upper:
        # High confidence — pass
        return {"validation_score": similarity}
    elif similarity < lower:
        # Low confidence — fail (will trigger correction loop)
        return {"validation_score": similarity}
    else:
        # Gray zone — invoke LLM-Judge
        judge_agrees = llm_judge(snippet_image, state["draft_answer"])
        if judge_agrees:
            # Judge agrees — boost score above threshold
            return {"validation_score": max(similarity, upper)}
        else:
            # Judge disagrees — force fail
            return {"validation_score": lower - 0.01}
