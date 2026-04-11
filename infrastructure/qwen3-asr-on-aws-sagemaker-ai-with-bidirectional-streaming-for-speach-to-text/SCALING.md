# Scaling Concurrent Streams on Qwen3-ASR SageMaker

This document covers two paths to handling multiple simultaneous audio streams on the Qwen3-ASR SageMaker endpoint: horizontal scaling (SageMaker-native, no code changes) and an async engine rewrite (code changes to vLLM integration for single-GPU concurrency).

---

## Table of Contents

- [The Concurrency Problem](#the-concurrency-problem)
- [Path A: Horizontal Scaling with SageMaker Auto Scaling](#path-a-horizontal-scaling-with-sagemaker-auto-scaling)
- [Path B: Async Engine Rewrite](#path-b-async-engine-rewrite)
- [Comparison](#comparison)
- [Recommendation](#recommendation)

---

## The Concurrency Problem

### Why the Current Architecture Serializes Inference

The current deployment uses vLLM's synchronous `LLM` class for inference. The call chain looks like this:

```
WebSocket handler (async)
    ↓
asyncio.to_thread(asr.streaming_transcribe, pcm, state)
    ↓
asr.streaming_transcribe()                          ← runs in thread pool
    ↓
self.model.generate([inp], sampling_params=...)     ← vLLM LLM.generate(), blocks thread
    ↓
GPU inference (synchronous, holds internal lock)
```

`asyncio.to_thread()` moves the blocking call off the event loop so the WebSocket handler can still accept new connections and respond to health checks. But vLLM's `LLM` class internally serializes `generate()` calls — if two threads call it concurrently, one waits for the other to finish.

With two concurrent streams, inference interleaves like this:

```
Stream A:  [──inference──]                [──inference──]
Stream B:                 [──inference──]                [──inference──]
GPU:       ████████████████████████████████████████████████████████████
                                                           time →
```

Each stream's per-chunk latency roughly doubles because it waits for the other stream's inference to complete. With N concurrent streams, each stream's latency scales by ~N×.

### The Re-Inference Cost

Qwen3-ASR's streaming works by re-feeding **all accumulated audio** to the model on every chunk (see [GUIDE.md, "How Streaming Inference Actually Works"](GUIDE.md#how-streaming-inference-actually-works)). A 30-second stream at chunk 15 processes all 30 seconds of audio, not just the latest 2-second chunk. This means:

- GPU compute per chunk grows linearly with stream duration
- Concurrent streams multiply this cost — two 30-second streams alternating chunks each experience ~60 seconds of GPU compute per chunk cycle
- KV cache usage also scales with accumulated audio length

This re-inference pattern makes single-GPU concurrency inherently expensive for long audio streams.

---

## Path A: Horizontal Scaling with SageMaker Auto Scaling

### How It Works

SageMaker can run multiple instances behind a single endpoint. Each instance has its own GPU, its own model copy, and its own container. SageMaker's router distributes incoming connections across instances:

```
                         ┌─ Instance 1 (L4 GPU) ─ Stream A
Client A ──►             │
              Endpoint ──┤
Client B ──►             │
                         └─ Instance 2 (L4 GPU) ─ Stream B
```

Each stream gets a dedicated GPU with no contention. Latency is identical to the single-stream case.

### Static Scaling

For a fixed number of concurrent streams, set `InitialInstanceCount` in `deploy.py`:

```python
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 4,       # ← supports 4 concurrent streams
        "InstanceType": "ml.g6.xlarge",
    }],
)
```

Or when creating/updating via CLI:

```bash
aws sagemaker create-endpoint-config \
    --endpoint-config-name qwen3-asr-bidi-streaming-config \
    --production-variants '[{
        "VariantName": "AllTraffic",
        "ModelName": "qwen3-asr-bidi-streaming-model",
        "InitialInstanceCount": 4,
        "InstanceType": "ml.g6.xlarge",
        "InitialVariantWeight": 1.0
    }]'
```

### Auto Scaling

For dynamic scaling based on demand, use [Application Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html). This involves two steps: registering the endpoint as a scalable target and attaching a scaling policy.

#### Step 1: Register the Scalable Target

Define the minimum and maximum instance count:

**AWS CLI:**
```bash
aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --min-capacity 1 \
    --max-capacity 8
```

**boto3:**
```python
import boto3

aas = boto3.client("application-autoscaling", region_name="us-east-1")

aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=8,
)
```

#### Step 2: Attach a Scaling Policy

Choose a metric that triggers scale-out and scale-in.

##### Option 1: Invocations Per Instance (Predefined Metric)

The only predefined auto-scaling metric available for traditional endpoints (without inference components). Scales based on average invocations per instance per minute:

```bash
cat > scaling-policy.json << 'EOF'
{
    "TargetValue": 1.0,
    "PredefinedMetricSpecification": {
        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
    },
    "ScaleInCooldown": 600,
    "ScaleOutCooldown": 300
}
EOF

aws application-autoscaling put-scaling-policy \
    --policy-name qwen3-asr-scaling-policy \
    --policy-type TargetTrackingScaling \
    --resource-id endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic \
    --service-namespace sagemaker \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

With `TargetValue: 1.0`, SageMaker scales out when the average exceeds 1 invocation per instance per minute and scales in when it drops below.

**Limitation for streaming**: A single bidirectional streaming connection is one long-lived invocation. The metric counts invocations per minute, so a stream lasting 60 seconds registers as ~1 invocation/min. This works for basic scaling (target 1.0 = one stream per instance), but the metric granularity is 1 minute, making it slow to react to burst traffic.

**boto3 equivalent:**
```python
aas.put_scaling_policy(
    PolicyName="qwen3-asr-scaling-policy",
    PolicyType="TargetTrackingScaling",
    ResourceId="endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic",
    ServiceNamespace="sagemaker",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 300,
    },
)
```

##### Option 2: Concurrent Requests Per Model (Custom Metric — Recommended for Streaming)

CloudWatch publishes a `ConcurrentRequestsPerModel` metric for all endpoints (including traditional ones without inference components) in the `AWS/SageMaker` namespace. This tracks the number of concurrent in-flight requests to each model and is a much better fit for streaming workloads where connections are long-lived.

The predefined high-resolution auto-scaling metrics (`SageMakerVariantConcurrentRequestsPerModelHighResolution` and `SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution`) are only available for [inference component-based endpoints](ENHANCEMENTS.md#migration-to-inference-components). For our traditional endpoint, we use a custom metric specification instead:

```bash
cat > scaling-policy.json << 'EOF'
{
    "TargetValue": 1.0,
    "CustomizedMetricSpecification": {
        "MetricName": "ConcurrentRequestsPerModel",
        "Namespace": "AWS/SageMaker",
        "Dimensions": [
            {"Name": "EndpointName", "Value": "qwen3-asr-bidi-streaming"},
            {"Name": "VariantName", "Value": "AllTraffic"}
        ],
        "Statistic": "Max"
    },
    "ScaleInCooldown": 600,
    "ScaleOutCooldown": 300
}
EOF

aws application-autoscaling put-scaling-policy \
    --policy-name qwen3-asr-scaling-policy \
    --policy-type TargetTrackingScaling \
    --resource-id endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic \
    --service-namespace sagemaker \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

With `TargetValue: 1.0` and `Statistic: Max`, SageMaker scales out when any instance has more than 1 concurrent connection, maintaining a 1:1 stream-to-instance ratio.

**boto3 equivalent:**
```python
aas.put_scaling_policy(
    PolicyName="qwen3-asr-scaling-policy",
    PolicyType="TargetTrackingScaling",
    ResourceId="endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic",
    ServiceNamespace="sagemaker",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1.0,
        "CustomizedMetricSpecification": {
            "MetricName": "ConcurrentRequestsPerModel",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [
                {"Name": "EndpointName", "Value": "qwen3-asr-bidi-streaming"},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            "Statistic": "Max",
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 300,
    },
)
```

> **Note**: To use the predefined high-resolution metrics (10-second granularity instead of 1-minute), migrate to an [inference component-based endpoint](ENHANCEMENTS.md#migration-to-inference-components).

#### Cooldown Tuning

| Parameter | Default | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| `ScaleOutCooldown` | 300s | 300s | 5 minutes prevents thrashing. New instances take 8-15 minutes to reach InService anyway, so shorter values don't help. |
| `ScaleInCooldown` | 300s | 600s | 10 minutes avoids premature scale-in during gaps between streams. Streaming workloads are often bursty. |

### Cost Model

Horizontal scaling costs scale linearly:

| Concurrent Streams | Instances | Cost (ml.g6.xlarge, us-east-1) |
|-------------------|-----------|-------------------------------|
| 1 | 1 | ~$1.07/hr |
| 4 | 4 | ~$4.28/hr |
| 8 | 8 | ~$8.56/hr |

Each instance loads its own copy of the model (~3.5 GB in GPU memory). The remaining ~20 GB per L4 GPU is used for KV cache.

### Tradeoffs

**Advantages:**
- Zero code changes
- True stream isolation — no latency degradation
- Battle-tested SageMaker infrastructure
- Each stream gets full GPU memory for KV cache

**Disadvantages:**
- Cold start on scale-out: 8-15 minutes for a new instance (container pull + model download + CUDA graph compilation)
- Cost scales linearly with concurrency
- Each instance is idle between streams (paying for GPU even when not processing audio)
- Scale-to-zero is possible (`MinCapacity: 0`) but the 8-15 minute cold start makes it impractical for real-time streaming

### Provisioned Concurrency Pattern

For predictable workloads, combine static and auto scaling:

```python
# Start with 2 always-on instances, auto-scale up to 8
aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=2,    # ← always-on baseline
    MaxCapacity=8,    # ← burst capacity
)
```

This gives you 2 streams with zero cold start and scales to 8 under load.

---

## Path B: Async Engine Rewrite

### What Changes

Replace vLLM's synchronous `LLM` class with `AsyncLLMEngine`, which supports true concurrent inference through continuous batching. Multiple `generate()` calls run in the same GPU forward pass instead of serializing.

```
Current (sync):                     Target (async):

Stream A → to_thread → LLM.generate   Stream A → await engine.generate ──┐
                         ↓                                                ├→ GPU batch
Stream B → to_thread → LLM.generate   Stream B → await engine.generate ──┘
                         ↓                         (continuous batching)
           (serialized)                            (concurrent)
```

### Why It's Non-Trivial

vLLM 0.14.0 provides two inference APIs:

| | `LLM` (current) | `AsyncLLMEngine` (target) |
|---|---|---|
| Level | High-level | Low-level |
| Init | `LLM(model=path, **kwargs)` | `EngineArgs(model=path, **kwargs)` → `AsyncLLMEngine.from_engine_args(args)` |
| Generate | `llm.generate(batch) → List[RequestOutput]` | `engine.generate(prompt, params, request_id) → AsyncIterator[RequestOutput]` |
| Batching | Pass list, get list back | Submit individual requests, engine batches internally |
| Lifecycle | Automatic | Manual start/shutdown |

There is **no `AsyncLLM` high-level wrapper** in vLLM 0.14.0. You must use the lower-level `AsyncLLMEngine` directly, which changes the calling pattern significantly.

### Scope of Changes

The changes are **not limited to the SageMaker adapter**. The streaming methods live in the upstream `qwen_asr` library:

| File | Lines | Method | Change |
|------|-------|--------|--------|
| `qwen_asr/inference/qwen3_asr.py` | 226-288 | `LLM()` class method | Add new `async_llm()` factory that creates `AsyncLLMEngine` |
| `qwen_asr/inference/qwen3_asr.py` | 521-537 | `_infer_asr_vllm()` | Make async, change from `llm.generate(batch)` to per-request `engine.generate()` |
| `qwen_asr/inference/qwen3_asr.py` | 657-765 | `streaming_transcribe()` | Make async, consume `AsyncIterator[RequestOutput]` |
| `qwen_asr/inference/qwen3_asr.py` | 767-830 | `finish_streaming_transcribe()` | Make async, same pattern |
| `sagemaker/app.py` | 200, 224 | WebSocket handler | Remove `asyncio.to_thread()`, call async methods directly |

### Detailed Changes

#### 1. New Async Factory (`qwen3_asr.py`)

```python
@classmethod
async def async_llm(cls, model: str, max_new_tokens=4096, **kwargs):
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import EngineArgs
    from vllm import SamplingParams

    engine_args = EngineArgs(model=model, **kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    processor = Qwen3ASRProcessor.from_pretrained(model, fix_mistral_regex=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    return cls(
        backend="vllm_async",
        model=engine,
        processor=processor,
        sampling_params=sampling_params,
    )
```

#### 2. Async `streaming_transcribe()` (`qwen3_asr.py`)

The core change — `self.model.generate()` returns an async iterator instead of a list:

```python
async def streaming_transcribe_async(self, pcm16k, state):
    # ... same audio buffering and prefix logic ...

    inp = {"prompt": prompt, "multi_modal_data": {"audio": [state.audio_accum]}}
    request_id = str(uuid.uuid4())

    # Consume the async iterator to get the final output
    final_output = None
    async for output in self.model.generate(inp, self.sampling_params, request_id):
        final_output = output

    gen_text = final_output.outputs[0].text

    # ... same state update logic ...
    return state
```

#### 3. Updated WebSocket Handler (`sagemaker/app.py`)

```python
# Before:
await asyncio.to_thread(asr.streaming_transcribe, pcm, state)

# After:
await asr.streaming_transcribe_async(pcm, state)
```

### Key Design Decisions

**Buffer vs. stream tokens?**
The simplest migration buffers the full generation (iterate the async iterator to completion, then read `.text`). Streaming individual tokens to the client is possible but adds complexity for marginal benefit — the transcription is already chunked by audio segments.

**Continuous batching behavior:**
When two streams submit `generate()` calls concurrently, `AsyncLLMEngine` batches them into a single GPU forward pass. This is vLLM's main concurrency advantage — it doesn't just interleave, it processes both requests simultaneously via PagedAttention.

**Backward compatibility:**
Keep the existing sync `LLM()` factory and methods unchanged. Add parallel `async_llm()` factory and `*_async()` method variants. This avoids breaking existing users of the library.

### GPU Memory Implications

Concurrent streams share GPU memory. On an L4 (24 GB):

| Component | Memory |
|-----------|--------|
| Model weights (1.7B, bfloat16) | ~3.9 GB |
| CUDA overhead | ~1.0 GB |
| KV cache (remaining) | ~19.1 GB |

vLLM's PagedAttention dynamically allocates KV cache. With the re-inference pattern, each stream's cache grows with accumulated audio. Two concurrent 30-second streams each need cache for ~30 seconds of audio context, reducing the effective cache available to each.

At some point, cache pressure causes vLLM to preempt requests (swap them to CPU), which degrades latency. The practical concurrent stream limit depends on audio duration and chunk size — short streams (< 30s) can likely support 3-4 concurrent on a single L4; longer streams fewer.

### Effort and Risk

| Phase | Work | Estimate |
|-------|------|----------|
| Async factory method | Create `async_llm()` factory with `AsyncLLMEngine` | 1-2 hours |
| Core method rewrites | Make `streaming_transcribe`, `finish_streaming_transcribe`, `_infer_asr_vllm` async | 8-12 hours |
| SageMaker integration | Update `app.py` to use async methods directly | 2-3 hours |
| Testing & validation | Verify output correctness, profile concurrency, test edge cases | 6-8 hours |
| Backward compatibility | Keep sync path, add feature detection | 2-3 hours |
| **Total** | | **~3-5 days** |

**Risks:**
- **vLLM async API stability**: `AsyncLLMEngine` is a lower-level API that may change across vLLM versions. Pinning `vllm==0.14.0` mitigates this but limits upgrades.
- **Multimodal input handling**: The `multi_modal_data` dict format may need adjustment for async engine. Needs testing.
- **Tokenizer thread safety**: The prefix rollback logic in `streaming_transcribe` calls `self.processor.tokenizer.encode()`/`.decode()`. HuggingFace tokenizers are generally thread-safe, but concurrent access patterns should be validated.
- **Upstream library changes**: Most changes are in `qwen_asr/inference/qwen3_asr.py`, not the SageMaker adapter. This means either forking the library or contributing upstream.

---

## Comparison

| | Path A: Horizontal Scaling | Path B: Async Engine |
|---|---|---|
| **Code changes** | None | ~3-5 days across `qwen_asr` + `sagemaker/app.py` |
| **Max concurrency** | Limited by `MaxCapacity` (SageMaker limit: typically 20+) | Limited by GPU memory (3-4 streams on L4 for short audio) |
| **Latency per stream** | Same as single-stream (dedicated GPU) | Increases modestly with concurrent streams (shared GPU) |
| **Cost** | Linear: N streams × $1.07/hr (g6.xlarge) | Sublinear: 1 instance serves multiple streams |
| **Cold start** | 8-15 min for new instances on scale-out | None (single instance already running) |
| **GPU utilization** | Low (one stream rarely saturates a GPU) | Higher (batched inference fills GPU better) |
| **Operational complexity** | Low (SageMaker-managed) | Medium (manage async lifecycle, handle cache pressure) |
| **Risk** | Low (standard SageMaker feature) | Medium (vLLM async API, upstream code changes) |
| **Best for** | Production workloads, predictable cost | Cost optimization at scale, high stream density |

---

## Recommendation

**Start with horizontal scaling (Path A).** It requires zero code changes, is production-ready immediately, and provides true stream isolation. Use the `ConcurrentRequestsPerModel` custom metric for scaling since it directly tracks active streaming connections (better than `InvocationsPerInstance` for long-lived connections).

A practical starting configuration:

```bash
# Register: 1 always-on instance, burst to 4
aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --min-capacity 1 \
    --max-capacity 4

# Scale on concurrent connections (target: 1 stream per instance)
aws application-autoscaling put-scaling-policy \
    --policy-name qwen3-asr-scaling-policy \
    --policy-type TargetTrackingScaling \
    --resource-id endpoint/qwen3-asr-bidi-streaming/variant/AllTraffic \
    --service-namespace sagemaker \
    --scalable-dimension sagemaker:variant:DesiredInstanceCount \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 1.0,
        "CustomizedMetricSpecification": {
            "MetricName": "ConcurrentRequestsPerModel",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [
                {"Name": "EndpointName", "Value": "qwen3-asr-bidi-streaming"},
                {"Name": "VariantName", "Value": "AllTraffic"}
            ],
            "Statistic": "Max"
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 300
    }'
```

**Consider the async engine rewrite (Path B)** if:
- Cost at scale becomes a concern (many concurrent streams × hours of usage)
- You need sub-instance-level concurrency (multiple short streams sharing one GPU)
- You're willing to maintain a fork of the `qwen_asr` streaming methods or contribute upstream

The two approaches are not mutually exclusive — you can use horizontal scaling today and add async engine support later for better GPU utilization. Auto scaling with the async engine would give you the best of both worlds: multiple streams per instance with more instances added under heavy load.
