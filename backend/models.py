# ============================================================
# models.py — Pydantic Schemas for the RAG API
# ============================================================

from pydantic import BaseModel, Field


class QueryPayload(BaseModel):
    """Incoming payload from the Chrome Extension frontend."""
    video_id: str = Field(..., description="YouTube video ID (e.g., 'dQw4w9WgXcQ')")
    timestamp: float = Field(..., description="Video timestamp in seconds when user drew the bounding box")
    bbox: list[float] = Field(..., description="Bounding box coordinates [x, y, width, height] relative to the video element")
    query: str = Field(..., description="User's question about the selected area")
    full_frame_b64: str = Field(..., description="Base64-encoded WebP screenshot of the full viewport")


class SSEEvent(BaseModel):
    """Server-Sent Event payload sent to the frontend."""
    status: str = Field(..., description="Event status: 'processing', 'complete', or 'error'")
    node: str | None = Field(None, description="Current LangGraph node name")
    thought: str | None = Field(None, description="Human-readable agent thought for UI display")
    answer: str | None = Field(None, description="Final synthesized answer (only on 'complete')")
    confidence: float | None = Field(None, description="SigLIP/Judge validation confidence score")
    message: str | None = Field(None, description="Error message (only on 'error')")
