#!/usr/bin/env python3
"""
Local integration test for the Qwen3-ASR SageMaker bidirectional streaming container.

Tests the WebSocket protocol, health check, and message flow using the mock mode
(no GPU required).

Usage:
    # Terminal 1 — start the server in mock mode:
    MOCK_MODE=1 python app.py serve

    # Terminal 2 — run the tests:
    python test_local.py
"""

import asyncio
import json
import struct
import sys

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None

try:
    import websockets
except ImportError:
    websockets = None

BASE_URL = "http://localhost:8080"
WS_URL = "ws://localhost:8080/invocations-bidirectional-stream"

passed = 0
failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  — {detail}")


# ── Test 1: Health check ─────────────────────────────────────────────────

async def test_ping():
    if httpx is None:
        report("ping (httpx)", False, "httpx not installed, using fallback")
        # fallback with urllib
        import urllib.request
        try:
            req = urllib.request.urlopen(f"{BASE_URL}/ping", timeout=2)
            report("ping", req.status == 200, f"status={req.status}")
        except Exception as e:
            report("ping", False, str(e))
        return

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/ping", timeout=2.0)
        report("ping returns 200", resp.status_code == 200, f"status={resp.status_code}")


# ── Test 2: Full streaming session (text frames) ────────────────────────

async def test_streaming_text_frames():
    """Send control messages as text frames, audio as binary — the direct WebSocket path."""
    async with websockets.connect(WS_URL) as ws:
        # Start session
        await ws.send(json.dumps({
            "type": "start",
            "language": None,
            "context": "",
            "chunk_size_sec": 1.0,
            "sample_format": "float32",
        }))

        # Send 3 chunks of 0.5 s silence each
        for i in range(3):
            chunk = np.zeros(8000, dtype=np.float32)  # 0.5s at 16kHz
            await ws.send(chunk.tobytes())
            resp = json.loads(await ws.recv())
            report(
                f"text-frames: chunk {i} transcription",
                resp.get("type") == "transcription",
                f"got type={resp.get('type')}",
            )

        # Finish
        await ws.send(json.dumps({"type": "finish"}))
        resp = json.loads(await ws.recv())
        report(
            "text-frames: final response",
            resp.get("type") == "final",
            f"got type={resp.get('type')}",
        )
        report(
            "text-frames: final has text",
            len(resp.get("text", "")) > 0,
            f"text={resp.get('text', '')!r}",
        )


# ── Test 3: Full streaming session (binary frames for control) ──────────

async def test_streaming_binary_frames():
    """
    Send control messages as binary frames (JSON encoded as bytes), mimicking
    how the SageMaker SDK delivers PayloadParts.
    """
    async with websockets.connect(WS_URL) as ws:
        # Start session — sent as binary
        start_msg = json.dumps({
            "type": "start",
            "language": "English",
            "sample_format": "float32",
        }).encode("utf-8")
        await ws.send(start_msg)

        # Send 2 chunks of 1 s audio
        for i in range(2):
            chunk = np.random.randn(16000).astype(np.float32) * 0.01
            await ws.send(chunk.tobytes())
            resp = json.loads(await ws.recv())
            report(
                f"binary-frames: chunk {i} response",
                resp.get("type") == "transcription",
                f"got type={resp.get('type')}",
            )

        # Finish — sent as binary
        finish_msg = json.dumps({"type": "finish"}).encode("utf-8")
        await ws.send(finish_msg)
        resp = json.loads(await ws.recv())
        report(
            "binary-frames: final response",
            resp.get("type") == "final",
            f"got type={resp.get('type')}",
        )


# ── Test 4: Error — audio before start ──────────────────────────────────

async def test_audio_before_start():
    """Sending audio before a start message should return an error."""
    async with websockets.connect(WS_URL) as ws:
        chunk = np.zeros(8000, dtype=np.float32).tobytes()
        await ws.send(chunk)
        resp = json.loads(await ws.recv())
        report(
            "error: audio before start",
            resp.get("type") == "error",
            f"got type={resp.get('type')}",
        )


# ── Test 5: int16 sample format ─────────────────────────────────────────

async def test_int16_format():
    """Test int16 PCM sample format."""
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({
            "type": "start",
            "sample_format": "int16",
        }))

        chunk = np.zeros(8000, dtype=np.int16).tobytes()
        await ws.send(chunk)
        resp = json.loads(await ws.recv())
        report(
            "int16: transcription received",
            resp.get("type") == "transcription",
            f"got type={resp.get('type')}",
        )

        await ws.send(json.dumps({"type": "finish"}))
        resp = json.loads(await ws.recv())
        report(
            "int16: final received",
            resp.get("type") == "final",
            f"got type={resp.get('type')}",
        )


# ── Runner ───────────────────────────────────────────────────────────────

async def main():
    print()
    print("=" * 60)
    print("  Qwen3-ASR SageMaker Container — Local Integration Tests")
    print("=" * 60)
    print()

    if websockets is None:
        print("ERROR: 'websockets' package is required. Install with: pip install websockets")
        sys.exit(1)

    print("[1/5] Health check")
    await test_ping()
    print()

    print("[2/5] Streaming session (text frames)")
    await test_streaming_text_frames()
    print()

    print("[3/5] Streaming session (binary control frames)")
    await test_streaming_binary_frames()
    print()

    print("[4/5] Error handling — audio before start")
    await test_audio_before_start()
    print()

    print("[5/5] int16 sample format")
    await test_int16_format()
    print()

    print("=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
