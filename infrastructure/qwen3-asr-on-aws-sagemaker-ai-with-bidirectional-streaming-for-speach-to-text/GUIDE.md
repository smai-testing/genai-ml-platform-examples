# Adapting Qwen3-ASR for SageMaker Bidirectional Streaming: A Complete Technical Guide

This guide explains everything involved in deploying [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — Alibaba's multilingual automatic speech recognition model — as a real-time streaming endpoint on Amazon SageMaker. It covers the Qwen3-ASR model architecture, SageMaker's bidirectional streaming feature, CUDA driver compatibility, and every design decision made along the way.

If you're new to any of these systems, read the whole thing. If you're experienced with one area, skip to the sections that are new to you.

---

## Table of Contents

- [Part 1: Understanding Qwen3-ASR](#part-1-understanding-qwen3-asr)
  - [What Is Qwen3-ASR?](#what-is-qwen3-asr)
  - [Model Architecture: Audio Encoder + LLM Decoder](#model-architecture-audio-encoder--llm-decoder)
  - [The Two Backends: Transformers vs vLLM](#the-two-backends-transformers-vs-vllm)
  - [Audio Processing Pipeline](#audio-processing-pipeline)
  - [The Streaming API](#the-streaming-api)
  - [How Streaming Inference Actually Works](#how-streaming-inference-actually-works)
- [Part 2: Understanding SageMaker Bidirectional Streaming](#part-2-understanding-sagemaker-bidirectional-streaming)
  - [What Is Bidirectional Streaming?](#what-is-bidirectional-streaming)
  - [How Traffic Flows](#how-traffic-flows)
  - [The Container Contract](#the-container-contract)
  - [The Client SDK](#the-client-sdk)
- [Part 3: Bridging the Two Systems](#part-3-bridging-the-two-systems)
  - [Design Overview](#design-overview)
  - [app.py: The WebSocket Server](#apppy-the-websocket-server)
  - [Message Protocol Design](#message-protocol-design)
  - [The Binary Frame Problem](#the-binary-frame-problem)
  - [Async Strategy: Why asyncio.to_thread()](#async-strategy-why-asyncioto_thread)
  - [Mock Mode for GPU-Free Testing](#mock-mode-for-gpu-free-testing)
- [Part 4: The CUDA Driver Compatibility Problem](#part-4-the-cuda-driver-compatibility-problem)
  - [The Problem](#the-problem)
  - [SageMaker GPU Instance Driver Versions](#sagemaker-gpu-instance-driver-versions)
  - [Why g5 Instances Don't Work](#why-g5-instances-dont-work)
  - [CUDA Forward Compatibility: The Solution](#cuda-forward-compatibility-the-solution)
  - [The entrypoint.sh Script](#the-entrypointsh-script)
  - [Instance Type Selection](#instance-type-selection)
- [Part 5: Container Image and Dockerfile](#part-5-container-image-and-dockerfile)
  - [Base Image Selection](#base-image-selection)
  - [Dependency Installation](#dependency-installation)
  - [The Bidirectional Streaming Label](#the-bidirectional-streaming-label)
  - [Building the Container](#building-the-container)
- [Part 6: Model Artifact and SageMaker Deployment](#part-6-model-artifact-and-sagemaker-deployment)
  - [How SageMaker Delivers Model Weights](#how-sagemaker-delivers-model-weights)
  - [Creating model.tar.gz](#creating-modeltargz)
  - [The Three-Step Deployment](#the-three-step-deployment)
  - [Startup Sequence](#startup-sequence)
- [Part 7: The Client and Credential Handling](#part-7-the-client-and-credential-handling)
  - [The Experimental HTTP/2 SDK](#the-experimental-http2-sdk)
  - [The Credential Problem](#the-credential-problem)
  - [Audio Streaming from Files](#audio-streaming-from-files)
- [Part 8: Lessons Learned and Pitfalls](#part-8-lessons-learned-and-pitfalls)

---

## Part 1: Understanding Qwen3-ASR

### What Is Qwen3-ASR?

Qwen3-ASR is a speech-to-text model from Alibaba's Qwen team. It transcribes audio in multiple languages (English, Chinese, Japanese, Korean, Cantonese, and many others) with state-of-the-art accuracy. The model comes in two sizes:

| Model | Parameters | WER (LibriSpeech clean) |
|-------|-----------|------------------------|
| Qwen3-ASR-0.6B | 600M | 2.06 |
| Qwen3-ASR-1.7B | 1.7B | 1.63 |

We use the 1.7B variant for better accuracy. It fits comfortably in a 24 GB GPU.

### Model Architecture: Audio Encoder + LLM Decoder

Qwen3-ASR is **not** a simple encoder-decoder model. It's a multimodal architecture that fuses a purpose-built audio encoder with a large language model:

```
                 ┌──────────────────────────────────────────┐
Raw Audio        │            Qwen3-ASR                      │
(16kHz PCM)      │                                           │
   │             │  ┌────────────────────────────────┐       │
   ▼             │  │    Audio Encoder (Whisper-like) │       │
Mel Spectrogram  │  │                                │       │
(128 mel bins)   │  │  Conv2d × 3 (stride 2 each)   │       │
   │             │  │      ↓                         │       │
   ▼             │  │  Sinusoidal Position Embedding │       │     Output:
Feature Map      │  │      ↓                         │       │     "language English"
   │             │  │  32 Transformer Encoder Layers  │       │     "<asr_text>"
   │             │  │  (d_model=1280, 20 attn heads) │       │     "Hello, world."
   │             │  │      ↓                         │       │
   │             │  │  Linear Projection (1280→3584) │──┐    │
   │             │  └────────────────────────────────┘  │    │
   │             │                                      │    │
   │             │  ┌────────────────────────────────┐  │    │
   │             │  │   Text Decoder (Qwen3 LLM)     │  │    │
   │             │  │                                │  │    │
   │             │  │  Token Embeddings              │  │    │
   │             │  │      ↓                         │  │    │
   │             │  │  [audio embeddings inserted]   │◄─┘    │
   │             │  │      ↓                         │       │
   │             │  │  24 Transformer Decoder Layers  │       │
   │             │  │  (RoPE, RMSNorm, causal mask)  │       │
   │             │  │      ↓                         │       │
   │             │  │  LM Head → vocab logits        │       │
   │             │  └────────────────────────────────┘       │
   │             └──────────────────────────────────────────┘
```

**Audio Encoder** (defined in `qwen_asr/core/transformers_backend/modeling_qwen3_asr.py:603-779`):
- Takes a 128-bin mel spectrogram as input
- Three strided convolutions reduce the time dimension by 8×
- 32 Transformer encoder layers process the sequence (similar to Whisper's architecture)
- Final linear projection maps from the encoder's 1280-dimensional space to the LLM's 3584-dimensional space
- Chunked processing (`n_window=100`, `n_window_infer=400`) prevents OOM on long audio

**Text Decoder** (Qwen3 LLM):
- A standard causal language model based on the Qwen3 architecture
- 24 Transformer decoder layers with rotary position embeddings (RoPE) and RMSNorm
- The audio encoder's output embeddings are inserted into the token sequence where `<|audio_token|>` placeholders appear in the prompt
- The model autoregressively generates text tokens: first the detected language tag, then the transcription

**Chat Template**: The model uses a chat-style prompt format:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|audio_token|><|audio_token|>...(N tokens)<|im_end|>
<|im_start|>assistant
```

The number of `<|audio_token|>` placeholders is computed from the audio length via `_get_feat_extract_output_lengths()`, which accounts for the convolution strides.

### The Two Backends: Transformers vs vLLM

Qwen3-ASR supports two inference backends:

**Transformers (HuggingFace)** — initialized via `Qwen3ASRModel.from_pretrained()`:
- Uses HuggingFace's `AutoModel` for inference
- Manages device and dtype manually
- No streaming support

**vLLM** — initialized via `Qwen3ASRModel.LLM()`:
- Uses [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference
- Supports PagedAttention, continuous batching, and CUDA graph compilation
- **Only backend that supports streaming** (this is why we use it)
- vLLM manages GPU memory internally via `gpu_memory_utilization` parameter

The backends are registered at import time in `qwen_asr/inference/qwen3_asr.py:21-54`:
```python
# Transformers backend
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

# vLLM backend (optional — only if vllm is installed)
try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model("Qwen3ASRForConditionalGeneration", ...)
except:
    pass
```

### Audio Processing Pipeline

Before audio reaches the model, it goes through a normalization chain (defined in `qwen_asr/inference/utils.py:135-201`):

```
Raw Input (file path, URL, base64, or numpy array)
    ↓
load_audio_any()     — Load from any source format
    ↓
to_mono()            — Convert stereo/multi-channel to mono
    ↓
librosa.resample()   — Resample to 16 kHz if needed
    ↓
float_range_normalize() — Clip to [-1.0, 1.0] range
    ↓
Result: 1D float32 numpy array, 16 kHz, mono
```

For streaming, the client is responsible for providing already-normalized audio: **16 kHz mono PCM in float32 or int16 format**. The server skips the normalization chain and feeds raw samples directly.

### The Streaming API

The streaming API lives in `qwen_asr/inference/qwen3_asr.py` and consists of three methods. Only the vLLM backend supports these.

#### 1. `init_streaming_state()` (lines 584-655)

Creates and returns an `ASRStreamingState` object that tracks the session:

```python
state = asr.init_streaming_state(
    context="",              # Optional text context/prompt
    language=None,           # None = auto-detect, or "English", "Chinese", etc.
    unfixed_chunk_num=2,     # First N chunks: no prefix constraining
    unfixed_token_num=5,     # Tokens to rollback for prefix generation
    chunk_size_sec=2.0,      # Audio chunk size in seconds
)
```

The `ASRStreamingState` dataclass (lines 77-128) holds:
- **Configuration**: `chunk_size_sec`, `chunk_size_samples`, `unfixed_chunk_num`, `unfixed_token_num`
- **Audio buffers**: `buffer` (un-processed samples), `audio_accum` (all audio seen so far)
- **Text state**: `prompt_raw` (the base prompt with chat template), `_raw_decoded` (raw model output), `language`, `text`
- **Counters**: `chunk_id` (current chunk index)

#### 2. `streaming_transcribe(pcm, state)` (lines 657-765)

Accepts new audio samples and returns updated transcription:

```python
state = asr.streaming_transcribe(pcm_chunk, state)
print(state.language, state.text)  # "English", "Hello, world..."
```

This method:
1. Appends incoming audio to `state.buffer`
2. While the buffer has enough samples for a full chunk (`chunk_size_samples`):
   - Extracts a chunk and appends it to `state.audio_accum`
   - Builds a prefix from previous output (see below)
   - Runs inference on **all accumulated audio** with the prefix
   - Updates `state.language` and `state.text`
   - Increments `state.chunk_id`

#### 3. `finish_streaming_transcribe(state)` (lines 767-830)

Flushes any remaining audio in the buffer (the "tail" shorter than a full chunk):

```python
state = asr.finish_streaming_transcribe(state)
print(state.text)  # Final complete transcription
```

### How Streaming Inference Actually Works

This is the most important concept to understand. Qwen3-ASR's streaming is **not** like a traditional streaming ASR system where the model maintains internal hidden states across chunks. Instead, it uses a **re-inference with prefix constraining** approach:

```
Chunk 1 arrives (2 seconds of audio):
  → Model sees: [audio_chunk_1]
  → Model generates: "Hello"
  → state.text = "Hello"

Chunk 2 arrives (2 more seconds):
  → Model sees: [audio_chunk_1 + audio_chunk_2]      ← ALL audio re-fed
  → Prefix constraint: "Hel"                          ← rollback 5 tokens from "Hello"
  → Model generates: "Hello, world"
  → state.text = "Hello, world"

Chunk 3 arrives:
  → Model sees: [audio_chunk_1 + audio_chunk_2 + audio_chunk_3]
  → Prefix constraint: "Hello, wo"                    ← rollback 5 tokens from "Hello, world"
  → Model generates: "Hello, world. How are"
  → state.text = "Hello, world. How are"

finish() called:
  → Model sees: [all audio including tail]
  → Final transcription: "Hello, world. How are you?"
```

**Why re-feed all audio?** The model doesn't have recurrent states. Each inference call is independent. The only way to give the model context about earlier audio is to include it.

**Why prefix constraining?** Without it, each chunk would generate a slightly different transcription of the earlier audio, causing the text to "jump around" between chunks. By forcing the model to start its output with the first N tokens from the previous output (minus a small rollback), the transcription remains stable.

**The `unfixed_chunk_num` parameter** controls how many initial chunks run without prefix constraining. The first few chunks have very little audio, so prefix constraining could over-constrain the output and produce errors. Default is 2-4 chunks.

**The `unfixed_token_num` parameter** controls how many tokens are rolled back from the end of the previous output before creating the prefix. This gives the model room to correct its ending. If the previous output was "Hello, worl" (incomplete word), rolling back 5 tokens lets the model fix it to "Hello, world" on the next chunk. Default is 5.

**Performance implication**: Inference cost grows with accumulated audio length. A 60-second audio file will re-process all 60 seconds on every chunk. This is fine for real-time streaming (where chunks arrive at 1× speed) but means you shouldn't use arbitrarily large chunk sizes.

---

## Part 2: Understanding SageMaker Bidirectional Streaming

### What Is Bidirectional Streaming?

Amazon SageMaker's [bidirectional streaming](https://aws.amazon.com/blogs/machine-learning/introducing-bidirectional-streaming-for-real-time-inference-on-amazon-sagemaker-ai/) feature (launched late 2024) allows clients to send data to and receive data from a SageMaker endpoint simultaneously over a persistent connection. This is different from standard SageMaker inference, which is request-response:

| Feature | Standard Inference | Bidirectional Streaming |
|---------|-------------------|------------------------|
| Protocol | HTTP POST/response | HTTP/2 (client) → WebSocket (container) |
| Connection | One request, one response | Persistent, both directions |
| Latency | Full input → full output | Incremental send → incremental receive |
| Use case | Batch processing | Real-time audio/video/chat |

### How Traffic Flows

```
 Your Code                   AWS Infrastructure                    Your Container
 ────────                   ──────────────────                    ──────────────

 HTTP/2 Client ──────►  SageMaker Bidi Router  ──────►  WebSocket Server
 (port 8443)              (managed by AWS)               (port 8080)
                                                          │
 Sends PayloadParts ───►  Routes to container  ──────►   │ receives binary frames
 (binary frames)          instance                        │
                                                          │
 Receives PayloadParts ◄──  Routes from container ◄────  │ sends text frames
 (binary frames)                                          │
```

Key details:
1. **Client side**: Uses HTTP/2 on port **8443** (not 443). The experimental `aws-sdk-sagemaker-runtime-http2` Python SDK handles this.
2. **SageMaker router**: Translates between the client's HTTP/2 stream and a WebSocket connection to your container.
3. **Container side**: Must serve a WebSocket endpoint at exactly `/invocations-bidirectional-stream` on port **8080**.

### The Container Contract

SageMaker requires your container to satisfy these requirements:

1. **HTTP health check**: `GET /ping` must return HTTP 200 within 2 seconds. SageMaker calls this to determine when the container is ready.

2. **WebSocket endpoint**: `WS /invocations-bidirectional-stream` on port 8080. This is the bidirectional streaming endpoint.

3. **Docker label**: The container image must include:
   ```dockerfile
   LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
   ```
   Without this label, SageMaker won't route bidirectional streaming traffic to your container.

4. **Port 8080**: SageMaker hardcodes this port for container communication.

### The Client SDK

SageMaker's bidirectional streaming uses an experimental Python SDK:

```bash
pip install aws-sdk-sagemaker-runtime-http2
```

This SDK is built on the Smithy framework and uses HTTP/2 to maintain a persistent bidirectional stream. The client sends `PayloadPart` objects (binary data) and receives them back. It's a lower-level SDK than boto3 — you're responsible for serializing/deserializing the data you send and receive.

The SDK connects to a special endpoint:
```
https://runtime.sagemaker.<region>.amazonaws.com:8443
```

Note the port 8443, not the standard HTTPS port 443.

---

## Part 3: Bridging the Two Systems

### Design Overview

The core challenge is bridging two very different interfaces:

| Qwen3-ASR Streaming API | SageMaker WebSocket Contract |
|--------------------------|------------------------------|
| Python function calls (`init_streaming_state`, `streaming_transcribe`, `finish_streaming_transcribe`) | WebSocket frames (binary and text) |
| Numpy arrays as input | Raw bytes as input |
| Synchronous (blocks on GPU) | Async (WebSocket event loop) |
| No network protocol | JSON control messages + binary audio |
| Single-threaded GPU access | Concurrent WebSocket connections |

Our `app.py` bridges these by:
1. Translating WebSocket messages to/from Qwen3-ASR API calls
2. Converting between raw bytes and numpy arrays
3. Using `asyncio.to_thread()` to run synchronous GPU inference without blocking the event loop
4. Managing per-connection session state

### app.py: The WebSocket Server

The server is a FastAPI application with two endpoints:

```python
app = FastAPI()

@app.get("/ping")
async def ping():
    return JSONResponse(content={}, status_code=200)

@app.websocket("/invocations-bidirectional-stream")
async def websocket_invoke(websocket: WebSocket):
    # ... bidirectional streaming logic
```

The WebSocket handler follows this lifecycle for each connection:

```python
async def websocket_invoke(websocket: WebSocket):
    await websocket.accept()
    state = None

    while True:
        message = await websocket.receive()

        control = _try_parse_control(text_data, binary_data)

        if control and control["type"] == "start":
            # Initialize Qwen3-ASR streaming state
            state = asr.init_streaming_state(...)

        elif control and control["type"] == "finish":
            # Flush remaining audio, send final transcription
            await asyncio.to_thread(asr.finish_streaming_transcribe, state)
            await websocket.send_text(json.dumps({"type": "final", ...}))
            break

        elif binary_data:
            # Decode PCM, run streaming inference, send partial result
            pcm = _decode_audio_bytes(binary_data, sample_format)
            await asyncio.to_thread(asr.streaming_transcribe, pcm, state)
            await websocket.send_text(json.dumps({"type": "transcription", ...}))
```

### Message Protocol Design

We designed a simple JSON + binary protocol:

**Client → Server:**
- `{"type": "start", "language": null, "chunk_size_sec": 2.0, "sample_format": "float32"}` — Start session
- Raw PCM bytes — Audio data
- `{"type": "finish"}` — End session

**Server → Client:**
- `{"type": "transcription", "language": "English", "text": "partial..."}` — After each chunk
- `{"type": "final", "language": "English", "text": "complete transcript"}` — After finish
- `{"type": "error", "message": "..."}` — On errors

This mirrors the structure of Qwen3-ASR's existing `demo_streaming.py` Flask endpoints (`/api/start`, `/api/chunk`, `/api/finish`) but consolidated into a single WebSocket connection.

### The Binary Frame Problem

This was a non-obvious issue we discovered during testing. In a standard WebSocket, there are two frame types:
- **Text frames**: For strings (JSON)
- **Binary frames**: For raw bytes (audio)

Our initial implementation expected JSON control messages as text frames and audio as binary frames. This works perfectly with a direct WebSocket client (e.g., Python's `websockets` library).

**The problem**: SageMaker's bidirectional streaming SDK sends **everything as binary frames**. The SDK wraps all data into `PayloadPart` objects with a `bytes_` field. So when the client sends `{"type": "start", ...}`, the container receives it as a binary WebSocket frame containing UTF-8-encoded JSON, not as a text frame.

**The solution**: The `_try_parse_control()` function attempts to parse JSON from both text and binary frames:

```python
def _try_parse_control(text_data, binary_data):
    raw_str = None
    if text_data:
        raw_str = text_data
    elif binary_data:
        try:
            raw_str = binary_data.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None  # Not valid UTF-8 → it's audio data

    if raw_str:
        try:
            obj = json.loads(raw_str)
            if isinstance(obj, dict) and "type" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    return None
```

The key insight: if a binary frame can be decoded as UTF-8 and parsed as JSON with a `"type"` field, it's a control message. Otherwise, it's audio data. This heuristic works because raw PCM audio bytes almost never happen to be valid UTF-8 JSON with a `"type"` field.

### Async Strategy: Why asyncio.to_thread()

FastAPI runs on an async event loop (uvicorn/asyncio). The WebSocket handler is an `async` function. But Qwen3-ASR's vLLM backend inference is **synchronous** — `LLM.generate()` blocks the calling thread until GPU inference completes.

If we called `asr.streaming_transcribe()` directly in the async handler, it would block the entire event loop, preventing other WebSocket connections from being serviced and breaking health check responses.

The solution is `asyncio.to_thread()`:

```python
await asyncio.to_thread(asr.streaming_transcribe, pcm, state)
```

This runs the synchronous function in a separate thread from Python's default thread pool, allowing the event loop to continue servicing other requests. When the GPU inference completes, the result is returned to the async handler.

**Caveat**: vLLM's `LLM` class is not designed for concurrent access from multiple threads. In practice, this works for a single-connection-at-a-time deployment. For multi-connection deployments, you would need vLLM's `AsyncLLMEngine` instead, but the current Qwen3-ASR streaming API uses the synchronous `LLM` class.

### Mock Mode for GPU-Free Testing

Testing the WebSocket protocol, frame parsing, and session lifecycle doesn't require a GPU. The `MOCK_MODE` environment variable replaces the real model with a `_MockASR` class that echoes audio statistics:

```python
class _MockASR:
    def init_streaming_state(self, **kwargs):
        return _MockStreamingState()

    def streaming_transcribe(self, pcm, state):
        state._samples_received += len(pcm)
        state.text = f"[mock] received {state._samples_received / 16000:.2f}s of audio"
        return state

    def finish_streaming_transcribe(self, state):
        state.text = f"[mock] final — {state._samples_received / 16000:.2f}s total audio"
        return state
```

This allows running the full test suite (`test_local.py`, 12 test cases) on any machine:
```bash
MOCK_MODE=1 python app.py serve      # Terminal 1
python test_local.py                  # Terminal 2
```

---

## Part 4: The CUDA Driver Compatibility Problem

This was the hardest problem we encountered and took multiple deployment attempts to solve.

### The Problem

The error `undefined symbol: cuTensorMapEncodeTiled` appeared in CloudWatch logs when the SageMaker endpoint tried to start. This cryptic message means: "the code was compiled for a newer CUDA version than what the GPU driver supports."

To understand why, you need to know how CUDA versioning works:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
│  (vLLM, PyTorch, flash-attention, etc.)                      │
│                                                              │
│  Compiled against CUDA Toolkit 12.8                          │
│  Uses APIs like cuTensorMapEncodeTiled (CUDA 12.0+)         │
├─────────────────────────────────────────────────────────────┤
│                   CUDA Runtime (libcudart.so)                │
│                   Version 12.8 (in container)                │
├─────────────────────────────────────────────────────────────┤
│                   CUDA Driver (libcuda.so)                    │
│                   Provided by host NVIDIA driver              │
│                   Must support APIs used by runtime           │
├─────────────────────────────────────────────────────────────┤
│                   NVIDIA Kernel Driver                        │
│                   Installed on the host machine               │
│                   Different per SageMaker instance type       │
└─────────────────────────────────────────────────────────────┘
```

The CUDA driver API is **backwards compatible within a major version** but **not across major versions**. A driver that supports CUDA 12.2 can run code compiled for CUDA 12.0-12.2, but not 12.8 (without the compatibility package). A driver that supports CUDA 11.4 cannot run CUDA 12.x code at all.

### SageMaker GPU Instance Driver Versions

SageMaker instances ship with fixed NVIDIA driver versions. You cannot upgrade the driver — it's baked into the AMI. These are the driver versions as of early 2025 (from the [SageMaker GPU driver documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html)):

| Instance Family | GPU | NVIDIA Driver | CUDA Version | Price (us-east-1) |
|-----------------|-----|---------------|--------------|-------------------|
| ml.g5.* | A10G (24GB) | **470** | **11.4** | ~$1.41/hr (xlarge) |
| ml.g6.* | L4 (24GB) | **535** | **12.2** | ~$1.07/hr (xlarge) |
| ml.g6e.* | L40S (48GB) | **535** | **12.2** | ~$2.17/hr (xlarge) |
| ml.p4d.* | A100 (40GB) | **470** | **11.4** | ~$37.69/hr |
| ml.p5.* | H100 (80GB) | **535** | **12.2** | ~$65.41/hr |
| ml.p5e.* | H100 (80GB) | **550** | **12.4** | varies |

### Why g5 Instances Don't Work

Our container is built on `nvidia/cuda:12.8.0-devel-ubuntu22.04`. vLLM 0.14.0's PyPI wheel is compiled for CUDA 12.9 (`cu129`). Both require a driver that supports CUDA 12.x APIs.

The `ml.g5.*` instances ship with NVIDIA driver **470**, which only supports CUDA **11.4**. The gap between CUDA 11 and CUDA 12 is a **major version boundary** — there is no compatibility bridge that can cross it. The `cuTensorMapEncodeTiled` API was introduced in CUDA 12.0 and simply doesn't exist in a driver that only knows CUDA 11.4.

We tried g5 first because it seemed like the obvious choice (A10G GPU, 24GB VRAM, widely available). The `cuTensorMapEncodeTiled` error only appears at runtime when the container tries to load vLLM's CUDA kernels, not during container build.

### CUDA Forward Compatibility: The Solution

NVIDIA provides a [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) package that bridges newer CUDA toolkit versions to older drivers **within the same major version**. This is exactly what we need for g6 instances: the host has driver 535 (CUDA 12.2) but our container needs CUDA 12.8.

The compatibility package (`cuda-compat-12-8`) provides a replacement `libcuda.so.1` at `/usr/local/cuda/compat/`. When loaded via `LD_LIBRARY_PATH` before the system's `libcuda.so`, it translates CUDA 12.8 API calls into calls the older driver understands:

```
Application (CUDA 12.8 APIs)
    ↓
/usr/local/cuda/compat/libcuda.so.1   ← Compat library (translates 12.8 → 12.2)
    ↓
Host NVIDIA driver 535 (CUDA 12.2)
    ↓
GPU hardware
```

This approach is [recommended by AWS](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html) for handling CUDA version mismatches in SageMaker containers.

### The entrypoint.sh Script

The `entrypoint.sh` script runs before the application starts and automatically enables CUDA forward compatibility when needed:

```bash
#!/bin/bash
set -euo pipefail

# Check if the compat package is installed
if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    # Get the compat package's max supported driver version
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d'.' -f 3-)

    # Get the host's actual driver version
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version)

    # If host driver is older → enable compat
    if verlt "$NVIDIA_DRIVER_VERSION" "$CUDA_COMPAT_MAX_DRIVER_VERSION"; then
        export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
    fi
fi

exec python app.py serve
```

The script:
1. Reads the host's NVIDIA driver version from `/proc/driver/nvidia/version`
2. Reads the compat package's maximum supported driver version from the `libcuda.so.1` symlink
3. If the host driver is older, prepends `/usr/local/cuda/compat/` to `LD_LIBRARY_PATH`
4. Falls back to enabling compat if versions can't be determined (safer than not enabling it)
5. Logs all diagnostics to stderr for CloudWatch

### Instance Type Selection

Based on our testing:

| Instance | Works? | Why |
|----------|--------|-----|
| `ml.g5.xlarge` (A10G) | **No** | Driver 470, CUDA 11.4 — major version gap |
| `ml.g6.xlarge` (L4) | **Yes** | Driver 535, CUDA 12.2 — compat bridges to 12.8 |
| `ml.g6.2xlarge` (L4) | **Yes** | Same GPU, more CPU/RAM |
| `ml.p5.xlarge` (H100) | **Yes** | Driver 535, CUDA 12.2 — compat bridges, overkill for 1.7B model |

**We recommend `ml.g6.xlarge`**: 24 GB L4 GPU, good performance, best price-to-capability ratio for the 1.7B model.

---

## Part 5: Container Image and Dockerfile

### Base Image Selection

We use `nvidia/cuda:12.8.0-devel-ubuntu22.04` as the base image. The `-devel` variant includes the full CUDA toolkit (compilers, headers, libraries) needed to compile vLLM's custom CUDA kernels and optional flash-attention.

Why CUDA 12.8 specifically:
- vLLM 0.14.0's PyPI wheels are compiled for CUDA 12.9 (`cu129`)
- CUDA 12.8 provides the runtime libraries needed
- vLLM's cu129 wheels are forward-compatible with CUDA 12.8 (minor version tolerance)
- We tried CUDA 12.4 first, but vLLM's wheels require 12.8+ runtime symbols

### Dependency Installation

The Dockerfile installs dependencies in a specific order to maximize Docker layer caching:

```dockerfile
# 1. System packages (rarely change)
RUN apt update && apt install -y python3 python3-pip git git-lfs libsndfile1 ffmpeg ...

# 2. CMake (for building native extensions)
RUN wget cmake-3.26.1 && install

# 3. Python packages (change when bumping versions)
RUN pip3 install -U "qwen-asr[vllm]" fastapi "uvicorn[standard]" websockets

# 4. Optional flash-attention (slow build, cached separately)
RUN if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then pip3 install flash-attn; fi

# 5. CUDA compat package (must match base CUDA version)
RUN apt-get install -y cuda-compat-12-8
```

The `qwen-asr[vllm]` package install pulls in:
- `qwen-asr`: The core Qwen3-ASR library
- `vllm`: The vLLM inference engine
- `torch`: PyTorch (CUDA-enabled)
- `transformers`: HuggingFace Transformers
- `librosa`, `soundfile`: Audio processing

The final image is ~14 GB, dominated by the CUDA toolkit, PyTorch, and vLLM.

### The Bidirectional Streaming Label

This single line is **required** for SageMaker to route bidirectional streaming traffic to the container:

```dockerfile
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
```

Without it, SageMaker treats the container as a standard inference container and never establishes a WebSocket connection. The `/invocations-bidirectional-stream` endpoint simply never receives any requests. There's no error message — it just doesn't work. This is easy to miss.

### Building the Container

Two options:

**Local Docker build** (`build_and_push.sh`):
```bash
docker build --platform linux/amd64 -t qwen3-asr-sagemaker .
```
The `--platform linux/amd64` flag is critical on Apple Silicon Macs, which default to `arm64`. SageMaker runs `x86_64` instances. Building under emulation is slow (~30-60 minutes).

**AWS CodeBuild** (`buildspec.yml`):
```yaml
build:
  commands:
    - docker build --build-arg BUNDLE_FLASH_ATTENTION=false -t $ECR_REPO_NAME .
```
CodeBuild runs native `x86_64` builds and is significantly faster. We use `BUILD_GENERAL1_LARGE` compute for adequate memory and CPU. Flash-attention is disabled (`BUNDLE_FLASH_ATTENTION=false`) to reduce build time — it's optional and the model works without it.

---

## Part 6: Model Artifact and SageMaker Deployment

### How SageMaker Delivers Model Weights

SageMaker has a specific model for getting model weights into a container. You don't bake the weights into the Docker image (that would make it 20+ GB and very slow to deploy). Instead:

1. You upload a `model.tar.gz` to S3
2. When creating the SageMaker model, you specify the S3 URI
3. SageMaker downloads and extracts it to `/opt/ml/model/` **before** starting the container
4. Your application reads from `/opt/ml/model/`

```
S3: s3://bucket/qwen3-asr/model.tar.gz
                    ↓
    SageMaker downloads & extracts
                    ↓
Container: /opt/ml/model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── ... (all model files)
```

### Creating model.tar.gz

This step has a subtle but critical detail: **the tarball must be created from inside the model directory**, not including the directory itself as a parent folder.

```bash
# Download the model
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B

# CORRECT: tar from inside the directory
cd Qwen3-ASR-1.7B
tar -czf ../model.tar.gz .

# WRONG: this creates model.tar.gz → Qwen3-ASR-1.7B/config.json
# SageMaker extracts to /opt/ml/model/Qwen3-ASR-1.7B/config.json ← wrong path!
tar -czf model.tar.gz Qwen3-ASR-1.7B/
```

If you create the tarball incorrectly, SageMaker extracts it to `/opt/ml/model/Qwen3-ASR-1.7B/` instead of `/opt/ml/model/`, and the model loading fails because `config.json` isn't where the code expects it.

The resulting `model.tar.gz` is ~3.5 GB for the 1.7B model.

### The Three-Step Deployment

SageMaker deployment involves three API calls:

**1. `create_model()`**: Associates the container image (ECR URI) with the model artifact (S3 URI) and an execution role (IAM):

```python
sm.create_model(
    ModelName="qwen3-asr-bidi-streaming-model",
    PrimaryContainer={
        "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/qwen3-asr-sagemaker:latest",
        "Mode": "SingleModel",
        "ModelDataUrl": "s3://bucket/qwen3-asr/model.tar.gz",
    },
    ExecutionRoleArn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
)
```

**2. `create_endpoint_config()`**: Specifies the instance type and how many instances:

```python
sm.create_endpoint_config(
    EndpointConfigName="qwen3-asr-bidi-streaming-config",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": "qwen3-asr-bidi-streaming-model",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.g6.xlarge",
    }],
)
```

**3. `create_endpoint()`**: Creates the actual endpoint, which triggers provisioning:

```python
sm.create_endpoint(
    EndpointName="qwen3-asr-bidi-streaming",
    EndpointConfigName="qwen3-asr-bidi-streaming-config",
)
```

### Startup Sequence

After `create_endpoint()`, SageMaker performs these steps (total: 8-15 minutes):

1. **Provision instance** (~2-3 min): Allocate an `ml.g6.xlarge` instance
2. **Pull container image** (~3-5 min): Download the ~14 GB image from ECR
3. **Download model** (~1-2 min): Download `model.tar.gz` from S3 and extract to `/opt/ml/model/`
4. **Start container**: Run `entrypoint.sh`
   - Detect host driver version
   - Enable CUDA compat if needed
   - Start `python app.py serve`
5. **Load model** (~30s): vLLM loads model weights into GPU memory
6. **CUDA graph compilation** (~1-2 min): vLLM compiles CUDA graphs for common sequence lengths
7. **Health check**: SageMaker calls `GET /ping` — once it returns 200, the endpoint is `InService`

You can monitor progress in CloudWatch logs at `/aws/sagemaker/Endpoints/<endpoint-name>`.

---

## Part 7: The Client and Credential Handling

### The Experimental HTTP/2 SDK

The `aws-sdk-sagemaker-runtime-http2` SDK is an experimental library from AWS for bidirectional streaming. It's fundamentally different from `boto3`:

- **Protocol**: HTTP/2 (not HTTP/1.1)
- **Port**: 8443 (not 443)
- **Authentication**: Uses Smithy framework's credential resolvers (not boto3's)
- **Data model**: `PayloadPart` objects with `bytes_` fields

Basic usage pattern:
```python
from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
from aws_sdk_sagemaker_runtime_http2.models import (
    InvokeEndpointWithBidirectionalStreamInput,
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)

# Connect
stream = await client.invoke_endpoint_with_bidirectional_stream(
    InvokeEndpointWithBidirectionalStreamInput(endpoint_name="my-endpoint")
)
output = await stream.await_output()

# Send data
await stream.input_stream.send(
    RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=data))
)

# Receive data
result = await output_stream.receive()
response_bytes = result.value.bytes_
```

### The Credential Problem

The HTTP/2 SDK uses Smithy's `EnvironmentCredentialsResolver`, which only reads credentials from environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`). It does **not** support:
- Named AWS profiles (`~/.aws/credentials`)
- AWS SSO
- Instance roles (EC2 metadata)
- Credential process

This is different from `boto3`, which supports all of these through its credential chain.

Our initial client hung indefinitely on "Connecting to endpoint" because the developer was using a named AWS profile, and the SDK couldn't find credentials.

**The solution** (learned from the [Deepgram SageMaker example](https://github.com/aws-samples/sagemaker-genai-hosting-examples/pull/157/files)): Use `boto3` to resolve credentials through the full credential chain, then export them as environment variables for the Smithy SDK:

```python
def resolve_credentials(region: str, profile: str | None = None):
    import boto3
    session = boto3.Session(region_name=region, profile_name=profile)
    credentials = session.get_credentials()
    frozen = credentials.get_frozen_credentials()

    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
```

This runs before initializing the HTTP/2 client, so by the time `EnvironmentCredentialsResolver` looks for credentials, they're in the environment.

### Audio Streaming from Files

The client simulates real-time streaming by loading an audio file, chunking it, and sending chunks with `asyncio.sleep()` delays:

```python
# Load audio file to 16kHz mono float32
pcm = librosa.load(path, sr=16000, mono=True)

# Send in 500ms chunks
chunk_samples = int(16000 * 0.5)  # 8000 samples = 500ms
for offset in range(0, len(pcm), chunk_samples):
    chunk = pcm[offset : offset + chunk_samples]
    await stream.input_stream.send(
        RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=chunk.tobytes()))
    )
    await asyncio.sleep(0.5)  # Simulate real-time pace
```

The `asyncio.sleep()` is important: without it, the client would blast all audio to the server instantly, which doesn't test the real-time streaming behavior and may overwhelm the server.

---

## Part 8: Lessons Learned and Pitfalls

### 1. g5 instances are tempting but don't work

The `ml.g5.xlarge` appears in every SageMaker GPU example and is widely available. Its A10G GPU has 24 GB VRAM, which is perfect for the 1.7B model. But its NVIDIA driver 470 (CUDA 11.4) is a dead end for any modern ML framework compiled against CUDA 12.x. There is no compatibility shim that crosses the CUDA 11 → 12 major version boundary.

**Always check the driver version** before choosing an instance type. The [SageMaker GPU driver docs](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html) list them.

### 2. The CUDA compat package must be explicitly installed

The `nvidia/cuda:12.8.0-devel-ubuntu22.04` base image does **not** include the CUDA forward compatibility package. You must install it separately:

```dockerfile
RUN apt-get update && apt-get install -y cuda-compat-12-8
```

The package name matches your CUDA version: `cuda-compat-12-8` for CUDA 12.8, `cuda-compat-12-4` for 12.4, etc.

### 3. SageMaker sends everything as binary frames

If you design your WebSocket protocol assuming JSON comes as text frames and audio as binary frames, it will work with a direct WebSocket client but fail with the SageMaker SDK. Always handle JSON in binary frames.

### 4. The bidirectional streaming Docker label is mandatory

```dockerfile
LABEL com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true
```

Without this, SageMaker silently ignores the WebSocket endpoint. No error, no warning — the endpoint just doesn't receive any streaming traffic.

### 5. model.tar.gz must be flat

Create the tarball from inside the model directory so that `config.json` is at the root of the archive, not nested in a subdirectory:

```bash
cd Qwen3-ASR-1.7B && tar -czf ../model.tar.gz .
```

### 6. The experimental HTTP/2 SDK doesn't use boto3 credentials

The `aws-sdk-sagemaker-runtime-http2` SDK uses Smithy's `EnvironmentCredentialsResolver`, which only reads environment variables. If your team uses AWS SSO, named profiles, or any other non-env-var credential method, you need the `resolve_credentials()` bridge function.

### 7. vLLM wheels are tied to specific CUDA versions

As of vLLM 0.14.0, the PyPI wheel is compiled for `cu129` (CUDA 12.9). You cannot simply change the base image to CUDA 12.4 and expect vLLM to work — it will fail with missing symbols. The base CUDA version must be compatible with the vLLM wheel's compilation target.

### 8. CodeBuild is better than local Docker on Apple Silicon

If you're developing on an Apple Silicon Mac (M1/M2/M3/M4), building `linux/amd64` Docker images requires QEMU emulation, which is extremely slow for large images. AWS CodeBuild provides native `x86_64` builds and is significantly faster. Use `buildspec.yml` with a CodeBuild project.

### 9. Flash-attention is optional

The Dockerfile optionally builds flash-attention (`BUNDLE_FLASH_ATTENTION=true`), which speeds up inference but adds significant build time. The model works without it. For faster iteration, disable it during development:

```bash
docker build --build-arg BUNDLE_FLASH_ATTENTION=false -t qwen3-asr-sagemaker .
```

### 10. The tokenizer regex warning is harmless

CloudWatch logs may show:
```
The tokenizer you are loading from '/opt/ml/model' with an incorrect regex pattern
```

This warning comes from vLLM's internal tokenizer loading, not from Qwen3-ASR's code (which already handles it with `fix_mistral_regex=True`). The warning doesn't affect transcription accuracy. Suppress it with:

```dockerfile
ENV TRANSFORMERS_VERBOSITY=error
```

---

## Appendix: File-by-File Reference

### app.py (327 lines)

The FastAPI WebSocket server. Key components:
- **Lines 52-60**: Configuration from environment variables
- **Lines 68-98**: Mock ASR class for GPU-free testing
- **Lines 101-118**: Model loading (real or mock)
- **Lines 127-135**: `/ping` health check
- **Lines 138-253**: `/invocations-bidirectional-stream` WebSocket handler
- **Lines 264-293**: `_try_parse_control()` — dual text/binary JSON parser
- **Lines 296-306**: `_decode_audio_bytes()` — PCM byte conversion

### Dockerfile (98 lines)

Container image definition. Key sections:
- **Line 12-13**: CUDA 12.8 base image
- **Lines 58-59**: `qwen-asr[vllm]` + FastAPI installation
- **Lines 72-74**: CUDA compat package installation
- **Line 84**: Bidirectional streaming label
- **Lines 90-96**: Environment variable defaults
- **Line 97**: Entrypoint

### entrypoint.sh (67 lines)

CUDA compat activation script. Key logic:
- **Lines 39-59**: Driver version comparison and `LD_LIBRARY_PATH` setting
- **Line 67**: `exec python app.py serve`

### client.py (216 lines)

Example bidirectional streaming client. Key functions:
- **Lines 53-83**: `resolve_credentials()` — boto3 → env var bridge
- **Lines 86-207**: `run()` — connect, stream, receive, close
- **Lines 136-154**: `process_responses()` — async response handler

### deploy.py (92 lines)

SageMaker deployment script. Creates model → endpoint config → endpoint.

### build_and_push.sh (54 lines)

Local Docker build and ECR push. Creates the ECR repo if needed.

### buildspec.yml (27 lines)

CodeBuild spec for remote builds. Disables flash-attention by default.

### test_local.py (230 lines)

Integration test suite (12 assertions across 5 test groups):
1. Health check (`/ping`)
2. Full session with text frames
3. Full session with binary control frames (SageMaker SDK path)
4. Error handling (audio before start)
5. int16 sample format support
