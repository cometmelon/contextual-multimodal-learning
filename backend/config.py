# ============================================================
# config.py — Central Configuration & Gemini API Key Rotation
# ============================================================
# Uses a round-robin key pool to distribute requests across
# multiple free-tier Gemini API keys, preventing 429 rate limits.
# ============================================================

import os
import itertools
from dataclasses import dataclass, field
from threading import Lock
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Key Pool ──────────────────────────────────────────────
# Multiple free-tier keys to rotate through and avoid rate limits.
# Each free-tier key allows ~15 RPM. With 4 keys = ~60 RPM effective.
_keys_env = os.environ.get("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [k.strip() for k in _keys_env.split(",") if k.strip()]

if not GEMINI_API_KEYS:
    print("⚠️ WARNING: No GEMINI_API_KEYS found in environment. Please set them in .env")
    GEMINI_API_KEYS = ["dummy_key"]


@dataclass
class KeyRotator:
    """Thread-safe round-robin API key rotator."""
    keys: list[str] = field(default_factory=lambda: GEMINI_API_KEYS)
    _cycle: itertools.cycle = field(init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self):
        self._cycle = itertools.cycle(self.keys)

    def next_key(self) -> str:
        with self._lock:
            return next(self._cycle)

    def get_client(self) -> genai.Client:
        """Returns a new Gemini client with the next rotated API key."""
        return genai.Client(api_key=self.next_key())

    def call_with_retry(self, model: str, contents, max_retries: int = 4):
        """
        Call Gemini with automatic key rotation on 429 errors.
        Tries each key in the pool before giving up.
        """
        import time
        last_error = None
        for attempt in range(max_retries):
            client = self.get_client()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                )
                return response
            except Exception as e:
                last_error = e
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 2 ** attempt  # 1, 2, 4, 8 seconds
                    print(f"  ⚠️ 429 rate limit hit, rotating key and waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
        raise last_error


# ── Singleton Rotator ─────────────────────────────────────────
key_rotator = KeyRotator()


# ── Model Configuration ───────────────────────────────────────
# Node 1 (Visual Label) + Node 5 (LLM-Judge): fast, lightweight
MODEL_FLASH = "gemini-2.5-flash"

# Node 4 (Heavy Synthesis): deep multimodal reasoning
MODEL_PRO = "gemini-2.5-flash" 

# ── Redis ─────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ── Backend ───────────────────────────────────────────────────
CORS_ORIGINS = [
    "https://www.youtube.com",
    "https://youtube.com",
    "chrome-extension://*",
]

# ── Guardrail Thresholds ──────────────────────────────────────
MAX_CORRECTION_ATTEMPTS = 3
TRANSCRIPT_WINDOW_SECONDS = 120  # +/- 60s from timestamp
