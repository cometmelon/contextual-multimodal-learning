# ============================================================
# agent/graph.py — LangGraph StateGraph Wiring & Compilation
# ============================================================
# Wires all nodes together with the self-correction loop.
# The conditional edge after the validator either passes, retries,
# or terminates with a best-guess warning.
# ============================================================

from langgraph.graph import StateGraph, END
from agent.graph_state import LeanAgentState
from agent.nodes import (
    node_visual_label,
    node_temporal_context,
    node_tool_router,
    node_synthesize,
    node_fusion_validator,
)
from config import MAX_CORRECTION_ATTEMPTS


def _routing_logic(state: LeanAgentState) -> str:
    """
    Conditional edge after the Fusion Validator.
    
    Decides whether to:
    - Accept the answer (validation passed)
    - Retry synthesis (validation failed, attempts remaining)
    - Give up with best-guess (max retries exceeded)
    """
    # Get dynamic thresholds for current content type
    from agent.validator import get_dynamic_thresholds
    upper, lower = get_dynamic_thresholds(state.get("visual_classification_label", ""))

    score = state.get("validation_score", 0.0)
    attempts = state.get("correction_attempts", 0)

    if score >= upper:
        # Validation passed
        return "accept"

    if attempts >= MAX_CORRECTION_ATTEMPTS:
        # Max retries exceeded — accept with warning
        return "accept"

    # Validation failed — trigger correction loop
    return "retry"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""

    workflow = StateGraph(LeanAgentState)

    # ── Register Nodes ────────────────────────────────────────
    workflow.add_node("node_visual_label", node_visual_label)
    workflow.add_node("node_temporal_context", node_temporal_context)
    workflow.add_node("node_tool_router", node_tool_router)
    workflow.add_node("node_synthesize", node_synthesize)
    workflow.add_node("node_fusion_validator", node_fusion_validator)

    # ── Wire Linear Edges ─────────────────────────────────────
    workflow.set_entry_point("node_visual_label")
    workflow.add_edge("node_visual_label", "node_temporal_context")
    workflow.add_edge("node_temporal_context", "node_tool_router")
    workflow.add_edge("node_tool_router", "node_synthesize")
    workflow.add_edge("node_synthesize", "node_fusion_validator")

    # ── Wire the Self-Correction Conditional Edge ─────────────
    workflow.add_conditional_edges(
        "node_fusion_validator",
        _routing_logic,
        {
            "accept": END,
            "retry": "node_synthesize",  # Loop back for correction
        },
    )

    return workflow.compile()
