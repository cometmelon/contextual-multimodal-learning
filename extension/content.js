// ============================================================
// content.js — Multimodal Video RAG Chrome Extension
// Injected on YouTube watch pages.
//
// Responsibilities:
//   1. Shadow DOM overlay injection with ResizeObserver
//   2. DRM pre-flight check
//   3. Bounding box drawing (mousedown/move/up)
//   4. UI occlusion mitigation (hide YT controls before capture)
//   5. OffscreenCanvas compression (4K → 1080p WebP ~300KB)
//   6. SSE listener for real-time agent thoughts
//   7. Result panel rendering
// ============================================================

(function () {
    "use strict";

    // ── Configuration ──────────────────────────────────────────
    let BACKEND_URL = "http://localhost:8000";
    const MAX_CAPTURE_WIDTH = 1920; // 1080p max dimension
    const WEBP_QUALITY = 0.7;

    // Load user-configured backend URL from extension settings
    chrome.storage.local.get(["backendUrl"], (result) => {
        if (result.backendUrl) BACKEND_URL = result.backendUrl;
    });

    // ── State ──────────────────────────────────────────────────
    let isSelectionMode = false;
    let isDrawing = false;
    let startX = 0,
        startY = 0;
    let currentRect = null;
    let shadowRoot = null;
    let overlayCanvas = null;
    let overlayCtx = null;
    let statusPanel = null;
    let resultPanel = null;
    let queryInput = null;
    let toggleBtn = null;
    let videoElement = null;
    let playerContainer = null;
    let resizeObserver = null;
    let eventSource = null;

    // ── Initialization ─────────────────────────────────────────
    function init() {
        // Wait for YouTube's video player to be ready
        const checkInterval = setInterval(() => {
            videoElement = document.querySelector("video.html5-main-video");
            playerContainer = document.querySelector("#movie_player");

            if (videoElement && playerContainer) {
                clearInterval(checkInterval);
                injectUI();
            }
        }, 500);
    }

    // ── UI Injection via Shadow DOM ────────────────────────────
    function injectUI() {
        // Create host element
        const host = document.createElement("div");
        host.id = "mrag-host";
        host.style.cssText =
            "position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9999;";
        playerContainer.style.position = "relative";
        playerContainer.appendChild(host);

        // Attach Shadow DOM for style isolation
        shadowRoot = host.attachShadow({ mode: "open" });

        // Inject shadow styles
        const style = document.createElement("style");
        style.textContent = getShadowStyles();
        shadowRoot.appendChild(style);

        // Create the transparent canvas overlay for drawing
        overlayCanvas = document.createElement("canvas");
        overlayCanvas.id = "mrag-overlay";
        shadowRoot.appendChild(overlayCanvas);

        // Create the status panel (SSE thoughts)
        statusPanel = document.createElement("div");
        statusPanel.id = "mrag-status";
        statusPanel.classList.add("hidden");
        shadowRoot.appendChild(statusPanel);

        // Create the result panel
        resultPanel = document.createElement("div");
        resultPanel.id = "mrag-result";
        resultPanel.classList.add("hidden");
        resultPanel.innerHTML = `
      <div class="result-header">
        <span class="result-title">AI Analysis</span>
        <button id="mrag-close-result" class="close-btn">&times;</button>
      </div>
      <div class="result-body"></div>
    `;
        shadowRoot.appendChild(resultPanel);

        // Create query input bar
        queryInput = document.createElement("div");
        queryInput.id = "mrag-query-bar";
        queryInput.classList.add("hidden");
        queryInput.innerHTML = `
      <input type="text" id="mrag-query-input" placeholder="Ask about the selected area... (e.g., What is this?)" />
      <button id="mrag-submit-query" class="submit-btn">
        <svg viewBox="0 0 24 24" width="18" height="18"><path fill="white" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    `;
        shadowRoot.appendChild(queryInput);

        // Create toggle button (in the light DOM, on the player)
        toggleBtn = document.createElement("button");
        toggleBtn.id = "mrag-toggle-btn";
        toggleBtn.classList.add("visible");
        toggleBtn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M15 3l2.3 2.3-2.89 2.87 1.42 1.42L18.7 6.7 21 9V3h-6zM3 9l2.3-2.3 2.87 2.89 1.42-1.42L6.7 5.3 9 3H3v6zm6 12l-2.3-2.3 2.89-2.87-1.42-1.42L5.3 17.3 3 15v6h6zm12-6l-2.3 2.3-2.87-2.89-1.42 1.42 2.89 2.87L15 21h6v-6z"/></svg>`;
        playerContainer.appendChild(toggleBtn);

        // Bind events
        bindEvents();

        // Setup ResizeObserver to track player size changes
        setupResizeObserver();

        // Initial canvas sizing
        resizeCanvas();
    }

    // ── Event Binding ──────────────────────────────────────────
    function bindEvents() {
        // Toggle button click
        toggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleSelectionMode();
        });

        // Canvas drawing events
        overlayCanvas.addEventListener("mousedown", onMouseDown);
        overlayCanvas.addEventListener("mousemove", onMouseMove);
        overlayCanvas.addEventListener("mouseup", onMouseUp);

        // Close result panel
        shadowRoot
            .getElementById("mrag-close-result")
            .addEventListener("click", () => {
                resultPanel.classList.add("hidden");
                // Reset host pointer-events so YouTube is interactive again
                const host = document.getElementById("mrag-host");
                if (host) host.style.pointerEvents = "none";
            });

        // Submit query
        shadowRoot
            .getElementById("mrag-submit-query")
            .addEventListener("click", submitQuery);

        // Keyboard isolation: Stop YouTube's global shortcuts from hijacking input
        // YouTube binds t=theater, f=fullscreen, k=pause, j/l=seek, etc. at document level.
        // We must kill event propagation on ALL keyboard events inside our input field.
        const inputEl = shadowRoot.getElementById("mrag-query-input");
        ["keydown", "keyup", "keypress"].forEach((eventType) => {
            inputEl.addEventListener(eventType, (e) => {
                e.stopPropagation(); // Prevent YouTube from seeing ANY keystrokes
                if (eventType === "keydown" && e.key === "Enter") {
                    submitQuery();
                }
            });
        });

        // Listen for background script messages (Alt+S shortcut)
        chrome.runtime.onMessage.addListener((msg) => {
            if (msg.action === "TOGGLE_SELECTION_MODE") {
                toggleSelectionMode();
            }
        });
    }

    // ── Selection Mode Toggle ──────────────────────────────────
    function toggleSelectionMode() {
        isSelectionMode = !isSelectionMode;
        overlayCanvas.style.pointerEvents = isSelectionMode ? "auto" : "none";
        overlayCanvas.style.cursor = isSelectionMode ? "crosshair" : "default";
        toggleBtn.classList.toggle("active", isSelectionMode);

        if (!isSelectionMode) {
            clearCanvas();
            queryInput.classList.add("hidden");
        }
    }

    // ── ResizeObserver ─────────────────────────────────────────
    function setupResizeObserver() {
        resizeObserver = new ResizeObserver(() => {
            resizeCanvas();
        });
        resizeObserver.observe(playerContainer);
    }

    function resizeCanvas() {
        const rect = playerContainer.getBoundingClientRect();
        overlayCanvas.width = rect.width;
        overlayCanvas.height = rect.height;
        overlayCanvas.style.width = rect.width + "px";
        overlayCanvas.style.height = rect.height + "px";
        overlayCtx = overlayCanvas.getContext("2d");
    }

    // ── Bounding Box Drawing ───────────────────────────────────
    function onMouseDown(e) {
        if (!isSelectionMode) return;
        e.preventDefault();
        e.stopPropagation();

        isDrawing = true;
        const canvasRect = overlayCanvas.getBoundingClientRect();
        startX = e.clientX - canvasRect.left;
        startY = e.clientY - canvasRect.top;

        // Hide previous results
        resultPanel.classList.add("hidden");
        queryInput.classList.add("hidden");
    }

    function onMouseMove(e) {
        if (!isDrawing) return;
        e.preventDefault();
        e.stopPropagation();

        const canvasRect = overlayCanvas.getBoundingClientRect();
        const currentX = e.clientX - canvasRect.left;
        const currentY = e.clientY - canvasRect.top;

        clearCanvas();
        drawSelectionRect(startX, startY, currentX - startX, currentY - startY);
    }

    function onMouseUp(e) {
        if (!isDrawing) return;
        e.preventDefault();
        e.stopPropagation();

        isDrawing = false;
        const canvasRect = overlayCanvas.getBoundingClientRect();
        const endX = e.clientX - canvasRect.left;
        const endY = e.clientY - canvasRect.top;

        const w = Math.abs(endX - startX);
        const h = Math.abs(endY - startY);

        // Ignore tiny accidental clicks (less than 10px)
        if (w < 10 || h < 10) {
            clearCanvas();
            return;
        }

        currentRect = {
            x: Math.min(startX, endX),
            y: Math.min(startY, endY),
            width: w,
            height: h,
        };

        // Redraw the final selection rectangle
        clearCanvas();
        drawSelectionRect(
            currentRect.x,
            currentRect.y,
            currentRect.width,
            currentRect.height
        );

        // Show query input bar
        queryInput.classList.remove("hidden");
        shadowRoot.getElementById("mrag-query-input").focus();
    }

    function drawSelectionRect(x, y, w, h) {
        overlayCtx.save();

        // Draw semi-transparent dark overlay on entire canvas
        overlayCtx.fillStyle = "rgba(0, 0, 0, 0.45)";
        overlayCtx.fillRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        // Clear the selected area (make it transparent → shows the video through)
        overlayCtx.clearRect(x, y, w, h);

        // Draw a glowing border around the selection
        overlayCtx.strokeStyle = "rgba(99, 102, 241, 0.9)";
        overlayCtx.lineWidth = 2.5;
        overlayCtx.shadowColor = "rgba(99, 102, 241, 0.6)";
        overlayCtx.shadowBlur = 12;
        overlayCtx.strokeRect(x, y, w, h);

        // Corner markers for visual fidelity
        const cornerLen = Math.min(16, w / 4, h / 4);
        overlayCtx.strokeStyle = "rgba(129, 140, 248, 1)";
        overlayCtx.lineWidth = 3;
        overlayCtx.shadowBlur = 0;

        // Top-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x, y + cornerLen);
        overlayCtx.lineTo(x, y);
        overlayCtx.lineTo(x + cornerLen, y);
        overlayCtx.stroke();

        // Top-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x + w - cornerLen, y);
        overlayCtx.lineTo(x + w, y);
        overlayCtx.lineTo(x + w, y + cornerLen);
        overlayCtx.stroke();

        // Bottom-left
        overlayCtx.beginPath();
        overlayCtx.moveTo(x, y + h - cornerLen);
        overlayCtx.lineTo(x, y + h);
        overlayCtx.lineTo(x + cornerLen, y + h);
        overlayCtx.stroke();

        // Bottom-right
        overlayCtx.beginPath();
        overlayCtx.moveTo(x + w - cornerLen, y + h);
        overlayCtx.lineTo(x + w, y + h);
        overlayCtx.lineTo(x + w, y + h - cornerLen);
        overlayCtx.stroke();

        overlayCtx.restore();
    }

    function clearCanvas() {
        if (overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }
    }

    // ── DRM Pre-Flight Check ───────────────────────────────────
    function isDRMProtected() {
        // If the video has no decoded frames, it's likely DRM-protected
        // and captureVisibleTab will return a black rectangle
        if (videoElement.webkitDecodedFrameCount !== undefined) {
            return videoElement.webkitDecodedFrameCount === 0 && videoElement.currentTime > 0;
        }
        return false;
    }

    // ── Submit Query Pipeline ──────────────────────────────────
    async function submitQuery() {
        const queryText =
            shadowRoot.getElementById("mrag-query-input").value.trim();
        if (!queryText || !currentRect) return;

        // DRM Pre-Flight
        if (isDRMProtected()) {
            showStatus("⚠️ DRM-protected content detected. Cannot capture frame.", "error");
            return;
        }

        // Disable further input
        queryInput.classList.add("hidden");

        // Show status panel
        showStatus("Capturing frame...", "processing");

        try {
            // Step 1: Hide YouTube UI controls to avoid occlusion
            const ytControls = document.querySelector(".ytp-chrome-bottom");
            const ytGradient = document.querySelector(".ytp-gradient-bottom");
            if (ytControls) ytControls.style.opacity = "0";
            if (ytGradient) ytGradient.style.opacity = "0";

            // Brief delay to let the repaint happen
            await sleep(100);

            // Step 2: Ask background worker to capture viewport
            const captureResponse = await new Promise((resolve) => {
                try {
                    chrome.runtime.sendMessage(
                        { action: "CAPTURE_VIEWPORT" },
                        (response) => {
                            if (chrome.runtime.lastError) {
                                resolve({ error: chrome.runtime.lastError.message });
                            } else {
                                resolve(response);
                            }
                        }
                    );
                } catch (err) {
                    // Extension context invalidated — user reloaded extension but not the page
                    resolve({ error: "Extension was updated. Please refresh this page (F5) and try again." });
                }
            });

            // Step 3: Restore YouTube UI controls
            if (ytControls) ytControls.style.opacity = "";
            if (ytGradient) ytGradient.style.opacity = "";

            if (captureResponse.error) {
                showStatus(`Capture failed: ${captureResponse.error}`, "error");
                return;
            }

            showStatus("Compressing frame...", "processing");

            // Step 4: Compress the capture via OffscreenCanvas
            const compressedB64 = await compressCapture(captureResponse.dataUrl);

            // Step 5: Calculate BBox coordinates relative to the video element
            const playerRect = playerContainer.getBoundingClientRect();
            const videoRect = videoElement.getBoundingClientRect();

            // Offset the bbox from player-relative to video-relative
            const offsetX = videoRect.left - playerRect.left;
            const offsetY = videoRect.top - playerRect.top;

            const bbox = [
                currentRect.x - offsetX,
                currentRect.y - offsetY,
                currentRect.width,
                currentRect.height,
            ];

            // Step 6: Extract video metadata
            const videoId = new URLSearchParams(window.location.search).get("v");
            const timestamp = videoElement.currentTime;

            // Step 7: Send payload & open SSE stream
            showStatus("Connecting to AI backend...", "processing");
            await sendToBackend({
                video_id: videoId,
                timestamp: timestamp,
                bbox: bbox,
                query: queryText,
                full_frame_b64: compressedB64,
            });
        } catch (err) {
            showStatus(`Error: ${err.message}`, "error");
        }
    }

    // ── OffscreenCanvas Compression ────────────────────────────
    async function compressCapture(dataUrl) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Determine scale factor for max 1080p width
                let targetWidth = img.width;
                let targetHeight = img.height;

                if (targetWidth > MAX_CAPTURE_WIDTH) {
                    const scale = MAX_CAPTURE_WIDTH / targetWidth;
                    targetWidth = MAX_CAPTURE_WIDTH;
                    targetHeight = Math.round(img.height * scale);
                }

                // Use OffscreenCanvas if available, else standard canvas
                let canvas, ctx;
                if (typeof OffscreenCanvas !== "undefined") {
                    canvas = new OffscreenCanvas(targetWidth, targetHeight);
                    ctx = canvas.getContext("2d");
                } else {
                    canvas = document.createElement("canvas");
                    canvas.width = targetWidth;
                    canvas.height = targetHeight;
                    ctx = canvas.getContext("2d");
                }

                ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

                // Encode as WebP with quality threshold
                if (canvas.convertToBlob) {
                    canvas
                        .convertToBlob({ type: "image/webp", quality: WEBP_QUALITY })
                        .then((blob) => blobToBase64(blob))
                        .then(resolve)
                        .catch(reject);
                } else {
                    resolve(canvas.toDataURL("image/webp", WEBP_QUALITY));
                }
            };
            img.onerror = () => reject(new Error("Failed to load captured image"));
            img.src = dataUrl;
        });
    }

    function blobToBase64(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    // ── Backend Communication (SSE) ────────────────────────────
    async function sendToBackend(payload) {
        // Close any previous SSE connections
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        console.log("[MRAG] Sending payload to backend:", {
            video_id: payload.video_id,
            timestamp: payload.timestamp,
            query: payload.query,
            bbox: payload.bbox,
            frame_length: payload.full_frame_b64?.length,
        });

        try {
            const response = await fetch(`${BACKEND_URL}/rag/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            console.log("[MRAG] Fetch response:", response.status, response.statusText);
            console.log("[MRAG] Response headers:", [...response.headers.entries()]);

            if (!response.ok) {
                throw new Error(`Backend returned ${response.status}`);
            }

            // Read the SSE stream from the response body
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";
            let chunkCount = 0;

            console.log("[MRAG] Starting to read SSE stream...");

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log("[MRAG] Stream ended (done=true). Total chunks:", chunkCount);
                    break;
                }

                chunkCount++;
                const chunk = decoder.decode(value, { stream: true });
                console.log(`[MRAG] Chunk #${chunkCount} (${chunk.length} chars):`, chunk.substring(0, 200));
                buffer += chunk;

                // Parse SSE events from the buffer
                const lines = buffer.split("\n");
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const data = line.slice(6).trim();
                        console.log("[MRAG] SSE data line:", data.substring(0, 200));

                        if (data === "[DONE]") {
                            console.log("[MRAG] Received [DONE] sentinel");
                            showStatus("Complete!", "success");
                            return;
                        }

                        try {
                            const event = JSON.parse(data);
                            console.log("[MRAG] Parsed SSE event:", event.status, event.node || "", event.thought || "");
                            handleSSEEvent(event);
                        } catch (parseErr) {
                            console.warn("[MRAG] Failed to parse SSE JSON:", parseErr.message, "| raw:", data.substring(0, 100));
                        }
                    }
                }
            }
        } catch (err) {
            console.error("[MRAG] Connection error:", err);
            showStatus(`Connection error: ${err.message}`, "error");
        }
    }

    function handleSSEEvent(event) {
        switch (event.status) {
            case "processing":
                showStatus(
                    event.thought || event.node || "Processing...",
                    "processing"
                );
                break;

            case "complete":
                console.log("[MRAG] Complete event received:", event);
                showStatus("✓ Analysis complete", "success");
                showResult(event.answer, event.confidence);

                // Exit selection mode
                isSelectionMode = false;
                overlayCanvas.style.pointerEvents = "none";
                overlayCanvas.style.cursor = "default";
                toggleBtn.classList.remove("active");
                clearCanvas();
                break;

            case "error":
                showStatus(`⚠️ ${event.message}`, "error");
                break;

            default:
                break;
        }
    }

    // ── Status Panel ───────────────────────────────────────────
    function showStatus(text, type) {
        statusPanel.classList.remove("hidden");
        statusPanel.className = `status-${type}`;
        statusPanel.innerHTML = `
      ${type === "processing" ? '<div class="spinner"></div>' : ""}
      <span>${text}</span>
    `;

        // Auto-hide on success/error after a delay
        if (type === "success" || type === "error") {
            setTimeout(() => {
                statusPanel.classList.add("hidden");
            }, 4000);
        }
    }

    // ── Result Panel ───────────────────────────────────────────
    function showResult(answer, confidence) {
        console.log("[MRAG] Showing result panel with answer:", answer?.substring(0, 100));
        resultPanel.classList.remove("hidden");

        // Ensure the host element allows pointer events for the result panel
        const host = document.getElementById("mrag-host");
        if (host) host.style.pointerEvents = "auto";

        const body = resultPanel.querySelector(".result-body");

        let confidenceBadge = "";
        if (confidence !== undefined && confidence !== null) {
            const color =
                confidence >= 0.75
                    ? "#22c55e"
                    : confidence >= 0.4
                        ? "#eab308"
                        : "#ef4444";
            confidenceBadge = `<div class="confidence-badge" style="color:${color}">Confidence: ${Math.round(confidence * 100)}%</div>`;
        }

        body.innerHTML = `
      ${confidenceBadge}
      <div class="answer-text">${formatAnswer(answer)}</div>
    `;
    }

    function formatAnswer(text) {
        if (!text) return "<em>No answer received.</em>";
        // Basic markdown-like formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
            .replace(/`(.*?)`/g, "<code>$1</code>")
            .replace(/\n/g, "<br>");
    }

    // ── Shadow DOM Styles ──────────────────────────────────────
    function getShadowStyles() {
        return `
      :host {
        all: initial;
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      }

      #mrag-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 9998;
        pointer-events: none;
        cursor: default;
      }

      /* ── Status Panel ────────────────────────── */
      #mrag-status {
        position: absolute;
        bottom: 72px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10001;
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 20px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 500;
        color: #e2e8f0;
        backdrop-filter: blur(16px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        white-space: nowrap;
      }

      #mrag-status.hidden { display: none; }

      .status-processing {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.4);
      }

      .status-success {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.4);
      }

      .status-error {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
      }

      .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255,255,255,0.2);
        border-top-color: #818cf8;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
      }

      @keyframes spin {
        to { transform: rotate(360deg); }
      }

      /* ── Query Input Bar ──────────────────────── */
      #mrag-query-bar {
        position: absolute;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10001;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 6px 6px 16px;
        width: min(500px, 85%);
        background: rgba(15, 15, 25, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        pointer-events: auto;
      }

      #mrag-query-bar.hidden { display: none; }

      #mrag-query-input {
        flex: 1;
        background: none;
        border: none;
        outline: none;
        color: #e2e8f0;
        font-size: 13.5px;
        font-family: inherit;
      }

      #mrag-query-input::placeholder {
        color: rgba(148, 163, 184, 0.6);
      }

      .submit-btn {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        border: none;
        background: rgba(99, 102, 241, 0.9);
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
        flex-shrink: 0;
      }

      .submit-btn:hover {
        background: rgba(129, 140, 248, 1);
        transform: scale(1.05);
      }

      /* ── Result Panel ────────────────────────── */
      #mrag-result {
        position: absolute;
        top: 12px;
        right: 60px;
        z-index: 10001;
        width: min(380px, 45%);
        max-height: 70%;
        overflow-y: auto;
        background: rgba(10, 10, 20, 0.9);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
        pointer-events: auto;
        animation: slideIn 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      }

      #mrag-result.hidden { display: none; }

      @keyframes slideIn {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
      }

      .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 16px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }

      .result-title {
        font-size: 14px;
        font-weight: 600;
        color: #c7d2fe;
        letter-spacing: 0.02em;
      }

      .close-btn {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        border: none;
        background: rgba(255,255,255,0.06);
        color: #94a3b8;
        font-size: 18px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s ease;
      }

      .close-btn:hover {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
      }

      .result-body {
        padding: 14px 16px 18px;
        font-size: 13.5px;
        line-height: 1.7;
        color: #cbd5e1;
      }

      .confidence-badge {
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
      }

      .answer-text code {
        background: rgba(99, 102, 241, 0.15);
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12.5px;
        color: #a5b4fc;
      }

      /* ── Scrollbar ────────────────────────────── */
      #mrag-result::-webkit-scrollbar {
        width: 5px;
      }
      #mrag-result::-webkit-scrollbar-track {
        background: transparent;
      }
      #mrag-result::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.3);
        border-radius: 4px;
      }
    `;
    }

    // ── Utility ────────────────────────────────────────────────
    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    // ── Bootstrap ──────────────────────────────────────────────
    // YouTube is an SPA; watch for navigation events
    let lastUrl = location.href;
    new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            if (location.href.includes("youtube.com/watch")) {
                // Clean up old injection
                const existingHost = document.getElementById("mrag-host");
                if (existingHost) existingHost.remove();
                const existingBtn = document.getElementById("mrag-toggle-btn");
                if (existingBtn) existingBtn.remove();
                init();
            }
        }
    }).observe(document.body, { childList: true, subtree: true });

    // Initial run
    if (document.readyState === "complete") {
        init();
    } else {
        window.addEventListener("load", init);
    }
})();
