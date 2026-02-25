# ============================================================
# main.py — FastAPI Backend for Multimodal Video RAG
# ============================================================
# Receives the dual-payload from the Chrome Extension, offloads
# images to Redis, and streams LangGraph agent thoughts via SSE.
# ============================================================

import json
import asyncio
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import CORS_ORIGINS
from models import QueryPayload, SSEEvent
from image_utils import crop_image, pil_to_bytes
from redis_client import store_image, generate_session_id, cleanup_session


# ── Lifespan (startup/shutdown) ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Multimodal Video RAG Backend starting...")
    yield
    print("🛑 Backend shutting down.")


app = FastAPI(
    title="Multimodal Video RAG",
    description="Agentic backend for the YouTube Chrome Extension",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── SSE Helper ────────────────────────────────────────────────
def sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    line = f"data: {json.dumps(data)}\n\n"
    print(f"  [SSE OUT] {json.dumps(data)[:200]}")
    return line


# ── Main Streaming Endpoint ──────────────────────────────────
@app.post("/rag/stream")
async def rag_stream(payload: QueryPayload):
    """
    Accepts the Chrome Extension payload, stores images in Redis,
    and returns an SSE stream of agent thoughts + final answer.
    """
    print(f"\n{'='*60}")
    print(f"[REQUEST] POST /rag/stream")
    print(f"  video_id: {payload.video_id}")
    print(f"  timestamp: {payload.timestamp}")
    print(f"  query: {payload.query}")
    print(f"  bbox: {payload.bbox}")
    print(f"  frame_b64 length: {len(payload.full_frame_b64)} chars")
    print(f"{'='*60}")

    async def event_generator():
        session_id = generate_session_id()
        print(f"[SESSION] {session_id}")

        try:
            # ── Step 1: Store full frame in Redis ──────────────
            print("[STEP 1] Storing full frame in Redis...")
            yield sse_event({"status": "processing", "thought": "Storing captured frame..."})

            from image_utils import decode_b64_to_pil
            full_frame_pil = decode_b64_to_pil(payload.full_frame_b64)
            print(f"  Full frame decoded: {full_frame_pil.size}")
            full_frame_bytes = pil_to_bytes(full_frame_pil)
            full_frame_ref = await store_image(session_id, "full", full_frame_bytes)
            print(f"  Stored as: {full_frame_ref} ({len(full_frame_bytes)} bytes)")

            # ── Step 2: Crop the snippet and store ─────────────
            print("[STEP 2] Cropping snippet...")
            yield sse_event({"status": "processing", "thought": "Extracting selected region..."})
            
            snippet_pil = crop_image(payload.full_frame_b64, payload.bbox)
            print(f"  Snippet cropped: {snippet_pil.size}")
            snippet_bytes = pil_to_bytes(snippet_pil)
            snippet_ref = await store_image(session_id, "snippet", snippet_bytes)
            print(f"  Stored as: {snippet_ref} ({len(snippet_bytes)} bytes)")

            # ── Step 3: Build lean state and run graph ─────────
            print("[STEP 3] Building graph...")
            yield sse_event({"status": "processing", "thought": "Initializing AI agent..."})

            from agent.graph import build_graph
            graph = build_graph()
            print("  Graph compiled successfully")

            initial_state = {
                "session_id": session_id,
                "video_id": payload.video_id,
                "timestamp": payload.timestamp,
                "query": payload.query,
                "bbox_coordinates": payload.bbox,
                "full_frame_ref": full_frame_ref,
                "snippet_ref": snippet_ref,
                "has_transcript": False,
                "transcript_context": "",
                "visual_classification_label": "",
                "tool_data": "",
                "draft_answer": "",
                "validation_score": 0.0,
                "correction_attempts": 0,
            }

            # ── Step 4: Stream agent execution ─────────────────
            print("[STEP 4] Starting graph execution...")
            accumulated_state = dict(initial_state)
            async for event in graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    print(f"\n  [NODE] {node_name}")
                    if isinstance(node_output, dict):
                        for key, val in node_output.items():
                            val_str = str(val)[:150]
                            print(f"    {key}: {val_str}")
                    
                    thought_map = {
                        "node_visual_label": "Categorizing visual context...",
                        "node_temporal_context": "Syncing transcript timelines...",
                        "node_tool_router": "Evaluating external knowledge sources...",
                        "node_synthesize": "Synthesizing answer from all context...",
                        "node_fusion_validator": "Running hallucination guardrail...",
                    }
                    thought = thought_map.get(node_name, f"Processing: {node_name}")
                    yield sse_event({
                        "status": "processing",
                        "node": node_name,
                        "thought": thought,
                    })

                    # Accumulate all node outputs into the running state
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)

            # ── Step 5: Send final result ──────────────────────
            print(f"\n[STEP 5] Graph complete!")
            print(f"  draft_answer: {accumulated_state.get('draft_answer', 'MISSING')[:200]}")
            print(f"  validation_score: {accumulated_state.get('validation_score', 'MISSING')}")
            print(f"  correction_attempts: {accumulated_state.get('correction_attempts', 'MISSING')}")
            
            if accumulated_state.get("draft_answer"):
                final_event = {
                    "status": "complete",
                    "answer": accumulated_state["draft_answer"],
                    "confidence": accumulated_state.get("validation_score", 0.0),
                }
                print(f"\n[FINAL SSE] Sending 'complete' event with answer ({len(accumulated_state['draft_answer'])} chars)")
                yield sse_event(final_event)
            else:
                print("[FINAL SSE] No draft_answer found — sending fallback")
                yield sse_event({
                    "status": "complete",
                    "answer": "I was unable to determine a confident answer for this visual context.",
                    "confidence": 0.0,
                })

        except Exception as e:
            print(f"\n[ERROR] {str(e)}")
            print(traceback.format_exc())
            yield sse_event({
                "status": "error",
                "message": f"Agent error: {str(e)}",
            })
        finally:
            # Cleanup Redis blobs
            await cleanup_session(session_id)
            print(f"[CLEANUP] Session {session_id} cleaned up")
            print("[DONE] Sending [DONE] sentinel")
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "multimodal-video-rag"}
