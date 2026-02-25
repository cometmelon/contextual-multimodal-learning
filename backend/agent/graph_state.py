# ============================================================
# agent/graph_state.py — LangGraph State Definition
# ============================================================
# The "Lean Agent State" — only lightweight pointers and text
# travel through the graph. Heavy image blobs stay in Redis.
# ============================================================

from typing import TypedDict


class LeanAgentState(TypedDict):
    # ── Base (from frontend payload) ──────────────────────────
    session_id: str
    video_id: str
    timestamp: float
    query: str

    # ── Context Derivations (populated by agent nodes) ────────
    has_transcript: bool
    transcript_context: str
    visual_classification_label: str
    tool_data: str  # Appended via Web Search Node

    # ── Lean Redis Pointers (NOT the actual image data) ───────
    full_frame_ref: str
    snippet_ref: str
    bbox_coordinates: list[float]  # [x, y, w, h] for spatial awareness

    # ── Orchestrator Mutations & Memory ───────────────────────
    draft_answer: str
    validation_score: float
    correction_attempts: int
