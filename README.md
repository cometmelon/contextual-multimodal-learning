# 🎯 Multimodal Video RAG

> **Draw a bounding box on any YouTube video frame and ask AI questions about what you see** — powered by temporal transcript context, multimodal reasoning, and anti-hallucination guardrails.

![Chrome Extension](https://img.shields.io/badge/Chrome-Extension%20MV3-4285F4?logo=googlechrome&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-FF6F00?logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini%20AI-Powered-8E75B2?logo=googlegemini&logoColor=white)

---

## ✨ Features

- **🖱️ Bounding Box Selection** — Draw a rectangle on any YouTube video frame to isolate a region of interest
- **🤖 5-Node Agentic Pipeline** — Visual labeling → Temporal transcript context → Tool routing → Multimodal synthesis → Fusion guardrail
- **📜 Transcript-Aware** — Automatically fetches and semantically searches the video transcript within a ±60s temporal window
- **🛡️ Anti-Hallucination Guardrails** — Tiered validation using SigLIP cosine similarity, dynamic thresholds, and an independent LLM-Judge
- **🔄 Self-Correction Loop** — Agent automatically retries synthesis if the guardrail detects inconsistencies (up to 3 attempts)
- **⚡ Real-Time Streaming** — Server-Sent Events (SSE) stream agent thoughts to the overlay in real time
- **🔑 API Key Rotation** — Round-robin rotation across multiple Gemini free-tier keys with automatic 429 retry

---

## 🏗️ Architecture

```
┌─────────────────────┐         SSE Stream          ┌──────────────────────┐
│   Chrome Extension   │ ◄──────────────────────────► │    FastAPI Backend    │
│                     │                              │                      │
│  • Shadow DOM UI    │    POST /rag/stream           │  • Image Storage     │
│  • BBox Drawing     │ ────────────────────────────► │  • Redis/fakeredis   │
│  • Frame Capture    │    {frame, bbox, query}       │  • Transcript Fetch  │
│  • Result Panel     │                              │                      │
└─────────────────────┘                              └──────────┬───────────┘
                                                                │
                                                    ┌───────────▼───────────┐
                                                    │   LangGraph Agent     │
                                                    │                       │
                                                    │  1. Visual Labeling   │
                                                    │  2. Temporal Context  │
                                                    │  3. Tool Router       │
                                                    │  4. Synthesis (Pro)   │
                                                    │  5. Fusion Guardrail  │
                                                    │     └─► Retry Loop    │
                                                    └───────────────────────┘
```

---

## 📁 Project Structure

```
contextual multimodal learning/
├── extension/                    # Chrome Extension (MV3)
│   ├── manifest.json             # Permissions, commands, content scripts
│   ├── background.js             # captureVisibleTab, keyboard shortcut relay
│   ├── content.js                # Shadow DOM UI, BBox, SSE client, result panel
│   ├── content.css               # Extension overlay styles
│   ├── popup.html / popup.js     # Settings popup (backend URL config)
│   └── icons/                    # Extension icons (16, 48, 128px)
│
├── backend/                      # FastAPI Backend
│   ├── main.py                   # SSE streaming endpoint, image pipeline
│   ├── config.py                 # API key rotation, model config, thresholds
│   ├── models.py                 # Pydantic request/response schemas
│   ├── image_utils.py            # Base64 decode, PIL crop, coordinate clamping
│   ├── redis_client.py           # Async Redis with fakeredis fallback
│   ├── transcript.py             # YouTube transcript fetch + semantic search
│   ├── requirements.txt          # Python dependencies
│   ├── .env.example              # Environment variable template
│   │
│   └── agent/                    # LangGraph Agent
│       ├── graph_state.py        # LeanAgentState TypedDict
│       ├── nodes.py              # 5 node functions (label, context, route, synth, validate)
│       ├── validator.py          # SigLIP + dynamic thresholds + LLM-Judge
│       └── graph.py              # StateGraph wiring + self-correction loop
│
├── System Implementation.md     # Detailed system design document
├── Review.md                    # Architecture review notes
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Google Chrome**
- **Gemini API Keys** ([Get free keys here](https://aistudio.google.com/apikey))

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your Gemini API keys (comma-separated)

# Start the server
python -m uvicorn main:app --reload --port 8000
```

### 2. Chrome Extension Setup

1. Open `chrome://extensions/` in Chrome
2. Enable **Developer mode** (toggle in top-right)
3. Click **Load unpacked** → select the `extension/` folder
4. Pin the extension from the toolbar

### 3. Usage

1. Navigate to any **YouTube video**
2. Press **Alt+S** or click the extension icon to enter selection mode
3. **Draw a bounding box** around any area of interest on the video
4. **Type your question** in the input field that appears
5. Press **Enter** — watch the AI think in real-time via SSE
6. View the **answer + confidence score** in the result panel

---

## ⚙️ Configuration

### Environment Variables (`.env`)

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEYS` | Comma-separated list of Gemini API keys | *Required* |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |

> **Note:** If Redis is not available, the backend automatically falls back to in-memory `fakeredis` for development.

### Model Configuration (`config.py`)

| Constant | Purpose | Default |
|---|---|---|
| `MODEL_FLASH` | Visual labeling, tool routing, LLM-Judge | `gemini-2.5-flash` |
| `MODEL_PRO` | Heavy multimodal synthesis | `gemini-2.5-flash` |
| `MAX_CORRECTION_ATTEMPTS` | Self-correction loop cap | `3` |
| `TRANSCRIPT_WINDOW_SECONDS` | Temporal context window (±seconds) | `120` |

---

## 🧪 Testing

### Test API Keys

```bash
cd backend
python test_keys.py
```

### Health Check

```bash
curl http://localhost:8000/health
```

---

## 🛡️ Tiered Fusion Guardrail

The anti-hallucination system uses three tiers:

| Tier | Method | Purpose |
|------|--------|---------|
| **Tier 1** | SigLIP cosine similarity | Deterministic math check between image and generated caption |
| **Tier 2** | Dynamic thresholds | Adapts based on content type — stricter for photos, relaxed for code/diagrams |
| **Tier 3** | LLM-Judge (Gemini Flash) | Independent verification for gray-zone scores |

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- [Google Gemini API](https://ai.google.dev/) — Multimodal AI backbone
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) — Vision-language similarity
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — Transcript extraction
