# ============================================================
# transcript.py — YouTube Transcript Fetching & Temporal Search
# ============================================================
# Wraps youtube-transcript-api in tenacity retry with exponential
# backoff for anti-scraping resilience. Implements temporal context
# windowing and basic semantic search within the transcript.
# ============================================================

from youtube_transcript_api import YouTubeTranscriptApi
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import TRANSCRIPT_WINDOW_SECONDS


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _fetch_raw_transcript(video_id: str) -> list[dict]:
    """
    Fetch the full transcript with retry logic.
    Retries up to 4 times with exponential backoff (1s, 2s, 4s, 8s).
    """
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


def fetch_transcript(video_id: str) -> tuple[bool, list[dict]]:
    """
    Safely fetch a YouTube video's transcript.
    
    Returns:
        (success: bool, transcript: list[dict])
        Each transcript entry: {"text": str, "start": float, "duration": float}
    """
    try:
        transcript = _fetch_raw_transcript(video_id)
        return True, transcript
    except Exception:
        return False, []


def get_temporal_window(
    transcript: list[dict],
    timestamp: float,
    window_seconds: int = TRANSCRIPT_WINDOW_SECONDS,
) -> list[dict]:
    """
    Extract a temporal window of transcript entries around the given timestamp.
    
    Args:
        transcript: Full transcript list
        timestamp: Center timestamp in seconds
        window_seconds: Total window size (default 120s = ±60s)
    
    Returns:
        Filtered list of transcript entries within the window
    """
    half_window = window_seconds / 2
    start_bound = max(0, timestamp - half_window)
    end_bound = timestamp + half_window

    return [
        entry for entry in transcript
        if start_bound <= entry["start"] <= end_bound
    ]


def semantic_search_transcript(
    transcript: list[dict],
    visual_label: str,
    query: str,
    timestamp: float,
    window_seconds: int = TRANSCRIPT_WINDOW_SECONDS,
) -> str:
    """
    Apply temporal windowing + keyword-based relevance filtering.
    
    First restricts to the temporal window, then scores each entry
    by keyword overlap with the visual label and query.
    
    Args:
        transcript: Full transcript list
        visual_label: Generated label from the Vision model (e.g., "Python dictionary")
        query: User's raw question
        timestamp: Video timestamp when user drew the bounding box
        window_seconds: Temporal window size
    
    Returns:
        Concatenated relevant transcript text
    """
    # Step 1: Temporal windowing
    window = get_temporal_window(transcript, timestamp, window_seconds)

    if not window:
        return ""

    # Step 2: Build keyword set from visual label + query
    keywords = set()
    for text in [visual_label.lower(), query.lower()]:
        # Simple tokenization — split on spaces and remove short words
        words = [w.strip(".,!?;:'\"()[]{}") for w in text.split()]
        keywords.update(w for w in words if len(w) > 2)

    # Step 3: Score each transcript entry by keyword overlap
    scored_entries = []
    for entry in window:
        entry_text_lower = entry["text"].lower()
        score = sum(1 for kw in keywords if kw in entry_text_lower)
        scored_entries.append((score, entry))

    # Step 4: Sort by relevance (score desc), then by timestamp (asc)
    scored_entries.sort(key=lambda x: (-x[0], x[1]["start"]))

    # Step 5: Take top entries (at least all entries, prioritized by relevance)
    # Always return the full window but reordered by relevance
    result_texts = [entry["text"] for _, entry in scored_entries]

    return " ".join(result_texts)
