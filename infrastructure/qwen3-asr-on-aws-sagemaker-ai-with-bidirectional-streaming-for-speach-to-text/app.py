# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
SageMaker Bidirectional Streaming container for Qwen3-ASR.

Exposes:
  GET  /ping                                → health check (HTTP 200)
  WS   /invocations-bidirectional-stream    → streaming ASR over WebSocket

Protocol (WebSocket frames — text or binary):
  The SageMaker SDK sends all data as PayloadPart bytes, which arrive as
  binary WebSocket frames.  The container also accepts native text frames
  for direct WebSocket clients.

  Client → Server:
    JSON (text or binary):  {"type":"start", "language":null, "context":"", "chunk_size_sec":2.0, "sample_format":"float32"}
    Binary (raw PCM):       audio bytes (16 kHz mono, float32 or int16)
    JSON (text or binary):  {"type":"finish"}

  Server → Client:
    Text:   {"type":"transcription", "language":"...", "text":"..."}
    Text:   {"type":"final",         "language":"...", "text":"..."}
    Text:   {"type":"error",         "message":"..."}

Run locally:
  python app.py serve
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qwen3-asr-sagemaker")

# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "32"))
MOCK_MODE = os.environ.get("MOCK_MODE", "").lower() in ("1", "true", "yes")

# Streaming defaults (can be overridden per-session via the start message)
DEFAULT_CHUNK_SIZE_SEC = float(os.environ.get("DEFAULT_CHUNK_SIZE_SEC", "2.0"))
DEFAULT_UNFIXED_CHUNK_NUM = int(os.environ.get("DEFAULT_UNFIXED_CHUNK_NUM", "4"))
DEFAULT_UNFIXED_TOKEN_NUM = int(os.environ.get("DEFAULT_UNFIXED_TOKEN_NUM", "5"))

# ---------------------------------------------------------------------------
# Global model singleton
# ---------------------------------------------------------------------------
asr = None  # Qwen3ASRModel or MockASR instance, loaded on startup


class _MockStreamingState:
    """Minimal streaming state for mock mode."""

    def __init__(self):
        self.language = ""
        self.text = ""
        self._samples_received = 0


class _MockASR:
    """
    Drop-in replacement for Qwen3ASRModel that echoes audio statistics
    instead of running real inference.  Useful for testing the WebSocket
    protocol on machines without a CUDA GPU.
    """

    def init_streaming_state(self, **kwargs):
        logger.info("[mock] init_streaming_state(%s)", kwargs)
        return _MockStreamingState()

    def streaming_transcribe(self, pcm: np.ndarray, state: _MockStreamingState):
        state._samples_received += len(pcm)
        secs = state._samples_received / 16000.0
        state.language = "English"
        state.text = f"[mock] received {secs:.2f}s of audio"
        return state

    def finish_streaming_transcribe(self, state: _MockStreamingState):
        secs = state._samples_received / 16000.0
        state.text = f"[mock] final — {secs:.2f}s total audio"
        return state


def load_model():
    """Load the Qwen3-ASR model using the vLLM backend (or mock for testing)."""
    global asr

    if MOCK_MODE:
        logger.info("MOCK_MODE enabled — using mock ASR (no GPU required)")
        asr = _MockASR()
        return

    from qwen_asr import Qwen3ASRModel

    logger.info("Loading model from %s (gpu_memory_utilization=%.2f) ...", MODEL_PATH, GPU_MEMORY_UTILIZATION)
    asr = Qwen3ASRModel.LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    logger.info("Model loaded successfully.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/ping")
async def ping():
    """SageMaker health-check endpoint. Must return 200 within 2 s."""
    return JSONResponse(content={}, status_code=200)


@app.websocket("/invocations-bidirectional-stream")
async def websocket_invoke(websocket: WebSocket):
    """
    Bidirectional streaming ASR session.

    Lifecycle:
      1. Client sends a JSON text frame {"type":"start", ...} to configure the session.
      2. Client streams binary audio frames (raw PCM bytes).
      3. Server sends back partial transcriptions after each processed chunk.
      4. Client sends {"type":"finish"} to end the session.
      5. Server sends the final transcription and closes.
    """
    await websocket.accept()
    state = None
    sample_format = "float32"  # "float32" or "int16"
    session_started = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.receive":
                # Extract payload — may arrive as text frame or binary frame.
                # The SageMaker SDK sends everything as binary PayloadParts,
                # so we must handle JSON control messages in binary frames too.
                text_data = message.get("text")
                binary_data = message.get("bytes")

                # Try to interpret as a JSON control message (from either frame type)
                control = _try_parse_control(text_data, binary_data)

                if control is not None:
                    msg_type = control.get("type", "")

                    if msg_type == "start":
                        if session_started:
                            await _send_error(websocket, "Session already started")
                            continue

                        language = control.get("language", None)
                        context = control.get("context", "")
                        chunk_size_sec = float(control.get("chunk_size_sec", DEFAULT_CHUNK_SIZE_SEC))
                        unfixed_chunk_num = int(control.get("unfixed_chunk_num", DEFAULT_UNFIXED_CHUNK_NUM))
                        unfixed_token_num = int(control.get("unfixed_token_num", DEFAULT_UNFIXED_TOKEN_NUM))
                        sample_format = control.get("sample_format", "float32")

                        state = asr.init_streaming_state(
                            context=context,
                            language=language,
                            unfixed_chunk_num=unfixed_chunk_num,
                            unfixed_token_num=unfixed_token_num,
                            chunk_size_sec=chunk_size_sec,
                        )
                        session_started = True
                        logger.info("Session started (language=%s, chunk=%.1fs, format=%s)",
                                    language, chunk_size_sec, sample_format)

                    elif msg_type == "finish":
                        if state is None:
                            await _send_error(websocket, "No active session to finish")
                            break

                        await asyncio.to_thread(asr.finish_streaming_transcribe, state)
                        await websocket.send_text(json.dumps({
                            "type": "final",
                            "language": state.language or "",
                            "text": state.text or "",
                        }))
                        logger.info("Session finished. Final: lang=%s, len=%d",
                                    state.language, len(state.text or ""))
                        break

                    else:
                        await _send_error(websocket, f"Unknown message type: {msg_type}")

                # Not a control message — treat as audio data
                elif binary_data:
                    if state is None:
                        await _send_error(websocket, "Send a 'start' message before streaming audio")
                        continue

                    pcm = _decode_audio_bytes(binary_data, sample_format)
                    if pcm is None:
                        await _send_error(websocket, "Invalid audio bytes (length not aligned to sample size)")
                        continue

                    await asyncio.to_thread(asr.streaming_transcribe, pcm, state)

                    await websocket.send_text(json.dumps({
                        "type": "transcription",
                        "language": state.language or "",
                        "text": state.text or "",
                    }))

                else:
                    # Empty frame or unrecognised text — skip
                    pass

            elif message["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await _send_error(websocket, str(e))
        except Exception:
            pass
    finally:
        # Clean up: flush remaining audio if session was active
        if state is not None and session_started:
            try:
                asr.finish_streaming_transcribe(state)
            except Exception:
                pass


async def _send_error(websocket: WebSocket, msg: str):
    """Send an error message to the client."""
    try:
        await websocket.send_text(json.dumps({"type": "error", "message": msg}))
    except Exception:
        pass


def _try_parse_control(text_data: Optional[str], binary_data: Optional[bytes]) -> Optional[dict]:
    """
    Attempt to parse a JSON control message from either a text or binary frame.

    The SageMaker bidirectional streaming SDK sends all PayloadParts as binary
    WebSocket frames, so JSON control messages (start/finish) arrive as binary.
    Direct WebSocket clients may send them as text frames instead.

    Returns the parsed dict if it contains a "type" field, otherwise None.
    """
    raw_str = None
    if text_data:
        raw_str = text_data
    elif binary_data:
        try:
            raw_str = binary_data.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    if raw_str is None:
        return None

    try:
        obj = json.loads(raw_str)
        if isinstance(obj, dict) and "type" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _decode_audio_bytes(raw: bytes, sample_format: str) -> Optional[np.ndarray]:
    """Convert raw PCM bytes to a float32 numpy array."""
    if sample_format == "int16":
        if len(raw) % 2 != 0:
            return None
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:  # float32
        if len(raw) % 4 != 0:
            return None
        pcm = np.frombuffer(raw, dtype=np.float32).copy()
    return pcm


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        main()
    else:
        print("Usage: python app.py serve")
        sys.exit(1)
