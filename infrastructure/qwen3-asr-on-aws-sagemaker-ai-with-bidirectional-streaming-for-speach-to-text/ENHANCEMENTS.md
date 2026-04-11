# Enhancements: Future Improvements for Qwen3-ASR on SageMaker

This document covers potential improvements to the current deployment, with a focus on migrating to SageMaker Inference Components for better scaling, resource management, and multi-model hosting.

---

## Table of Contents

- [Migration to Inference Components](#migration-to-inference-components)
  - [What Are Inference Components?](#what-are-inference-components)
  - [Why Migrate?](#why-migrate)
  - [Architecture Comparison](#architecture-comparison)
  - [What Changes in the Deployment](#what-changes-in-the-deployment)
  - [Step-by-Step Migration](#step-by-step-migration)
  - [Auto Scaling with Inference Components](#auto-scaling-with-inference-components)
  - [Container and Application Changes](#container-and-application-changes)
  - [Invocation Changes](#invocation-changes)
  - [Cleanup of the Old Deployment](#cleanup-of-the-old-deployment)
- [Connection Concurrency Guard](#connection-concurrency-guard)
- [Other Potential Enhancements](#other-potential-enhancements)

---

## Migration to Inference Components

### What Are Inference Components?

[Inference Components](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html) are a newer SageMaker hosting primitive that decouples the **model** from the **endpoint infrastructure**. Instead of baking the model into an endpoint configuration (the traditional approach), you create an endpoint with compute capacity and then deploy models onto it as independent "inference components."

Think of it as the difference between a dedicated server and a container platform:

```
Traditional Endpoint:               Inference Component Endpoint:
┌─────────────────────────┐         ┌─────────────────────────────────┐
│ Endpoint                │         │ Endpoint (compute pool)         │
│  └─ ProductionVariant   │         │  └─ ProductionVariant           │
│      ├─ Model           │         │      ├─ Instance 1 (g6.xlarge)  │
│      ├─ InstanceType    │         │      └─ Instance 2 (g6.xlarge)  │
│      └─ InstanceCount   │         │                                 │
│                         │         │  InferenceComponent A           │
│  (model + infra tightly │         │    ├─ Container image           │
│   coupled)              │         │    ├─ Model artifact            │
│                         │         │    ├─ Resource requirements      │
└─────────────────────────┘         │    └─ CopyCount: 2             │
                                    │                                 │
                                    │  InferenceComponent B (optional)│
                                    │    ├─ Different model            │
                                    │    └─ CopyCount: 1             │
                                    └─────────────────────────────────┘
```

### Why Migrate?

| Benefit | Traditional Endpoint | Inference Components |
|---------|---------------------|----------------------|
| **Auto-scaling granularity** | Instance count only | Model copies AND instance count independently |
| **Auto-scaling metrics** | `InvocationsPerInstance` (1-min granularity) | `ConcurrentRequestsPerCopyHighResolution` (10-second granularity) |
| **Scale to zero** | Possible but impractical (cold start) | Model copies scale to 0 while instance stays warm — faster recovery |
| **Multi-model hosting** | One model per endpoint | Multiple models share instances |
| **Resource specification** | Implicit (whole instance) | Explicit (CPU cores, memory, accelerators per component) |
| **Model updates** | Requires new endpoint config + rolling update | Update component independently, no endpoint downtime |
| **Instance scaling** | Manual or via auto-scaling | `ManagedInstanceScaling` — SageMaker handles instance provisioning automatically |

For our streaming ASR use case, the key advantages are:

1. **High-resolution scaling metrics**: `ConcurrentRequestsPerCopyHighResolution` emits every 10 seconds (vs 1 minute for traditional metrics), allowing much faster reaction to new streaming connections.

2. **Managed instance scaling**: SageMaker automatically provisions and deprovisions instances based on the inference components' resource demands, without manual auto-scaling configuration for the instances themselves.

3. **Independent model updates**: Deploy a new model version by updating the inference component. SageMaker handles the rollout without recreating the endpoint.

### Architecture Comparison

**Current (traditional):**
```
deploy.py:
  create_model()           ← ties container + S3 artifact + IAM role
  create_endpoint_config() ← ties model + instance type + instance count
  create_endpoint()        ← provisions everything

Scaling: application-autoscaling on DesiredInstanceCount
```

**Target (inference components):**
```
deploy.py:
  create_endpoint_config() ← defines instance type + managed scaling
  create_endpoint()        ← provisions compute pool
  create_inference_component() ← deploys model with resource requirements + copy count

Scaling:
  - Instance level: ManagedInstanceScaling (automatic)
  - Component level: application-autoscaling on DesiredCopyCount
```

### What Changes in the Deployment

The container image (`Dockerfile`, `app.py`, `entrypoint.sh`) does **not change**. The WebSocket server, health check, and bidirectional streaming label all work the same way. Only the deployment script and scaling configuration change.

| File | Changes |
|------|---------|
| `Dockerfile` | None |
| `app.py` | None |
| `entrypoint.sh` | None |
| `deploy.py` | Rewrite — replace `create_model()` with `create_inference_component()` |
| `client.py` | Minor — add `InferenceComponentName` to invocation if targeting specific component |
| Auto-scaling config | New — use `DesiredCopyCount` dimension and high-resolution metrics |

### Step-by-Step Migration

#### 1. Create an Endpoint Config with Managed Instance Scaling

The endpoint config no longer references a model. Instead, it defines a compute pool with managed scaling:

**boto3:**
```python
sm = boto3.client("sagemaker", region_name="us-east-1")

sm.create_endpoint_config(
    EndpointConfigName="qwen3-asr-ic-config",
    ExecutionRoleArn="arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole>",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "InstanceType": "ml.g6.xlarge",
        "InitialInstanceCount": 1,
        "ManagedInstanceScaling": {
            "Status": "ENABLED",
            "MinInstanceCount": 1,
            "MaxInstanceCount": 4,
        },
        # No ModelName — model is deployed via InferenceComponent
    }],
)
```

**CLI:**
```bash
aws sagemaker create-endpoint-config \
    --endpoint-config-name qwen3-asr-ic-config \
    --execution-role-arn arn:aws:iam::<ACCOUNT_ID>:role/<SageMakerExecutionRole> \
    --production-variants '[{
        "VariantName": "AllTraffic",
        "InstanceType": "ml.g6.xlarge",
        "InitialInstanceCount": 1,
        "ManagedInstanceScaling": {
            "Status": "ENABLED",
            "MinInstanceCount": 1,
            "MaxInstanceCount": 4
        }
    }]'
```

`ManagedInstanceScaling` tells SageMaker to automatically add or remove instances based on the deployed inference components' resource needs. You don't need to configure Application Auto Scaling for instance count — SageMaker handles it.

#### 2. Create the Endpoint

```python
sm.create_endpoint(
    EndpointName="qwen3-asr-bidi-streaming",
    EndpointConfigName="qwen3-asr-ic-config",
)
```

#### 3. Deploy the Model as an Inference Component

This replaces the old `create_model()` call. The inference component specifies the container, model artifact, resource requirements, and desired copy count:

**boto3:**
```python
sm.create_inference_component(
    InferenceComponentName="qwen3-asr-1.7b",
    EndpointName="qwen3-asr-bidi-streaming",
    VariantName="AllTraffic",
    Specification={
        "Container": {
            "Image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/qwen3-asr-sagemaker:latest",
            "ArtifactUrl": "s3://<BUCKET>/qwen3-asr/model.tar.gz",
        },
        "ComputeResourceRequirements": {
            "NumberOfAcceleratorDevicesRequired": 1,   # 1 GPU per copy
            "NumberOfCpuCoresRequired": 2,
            "MinMemoryRequiredInMb": 16384,            # 16 GB system memory
        },
    },
    RuntimeConfig={
        "CopyCount": 1,    # Start with 1 copy, auto-scale from here
    },
)
```

**CLI:**
```bash
aws sagemaker create-inference-component \
    --inference-component-name qwen3-asr-1.7b \
    --endpoint-name qwen3-asr-bidi-streaming \
    --variant-name AllTraffic \
    --specification '{
        "Container": {
            "Image": "<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/qwen3-asr-sagemaker:latest",
            "ArtifactUrl": "s3://<BUCKET>/qwen3-asr/model.tar.gz"
        },
        "ComputeResourceRequirements": {
            "NumberOfAcceleratorDevicesRequired": 1,
            "NumberOfCpuCoresRequired": 2,
            "MinMemoryRequiredInMb": 16384
        }
    }' \
    --runtime-config '{"CopyCount": 1}'
```

**Resource requirements explained:**
- `NumberOfAcceleratorDevicesRequired: 1` — each copy of the model needs one GPU. On `ml.g6.xlarge` (1 × L4 GPU), this means one copy per instance.
- `NumberOfCpuCoresRequired: 2` — reserve 2 CPU cores for audio processing and the WebSocket server.
- `MinMemoryRequiredInMb: 16384` — 16 GB system memory for model loading overhead, audio buffers, and vLLM's CPU-side allocations.

#### 4. Wait for the Inference Component to be Active

```python
# Poll until active
import time

while True:
    resp = sm.describe_inference_component(
        InferenceComponentName="qwen3-asr-1.7b"
    )
    status = resp["InferenceComponentStatus"]
    print(f"Status: {status}")
    if status == "InService":
        break
    elif status == "Failed":
        print(f"Failed: {resp.get('FailureReason', 'Unknown')}")
        break
    time.sleep(30)
```

### Auto Scaling with Inference Components

Inference components unlock the high-resolution predefined auto-scaling metrics that aren't available for traditional endpoints.

#### Register the Inference Component as a Scalable Target

The scalable dimension is `sagemaker:inference-component:DesiredCopyCount` (not `DesiredInstanceCount`):

```bash
aws application-autoscaling register-scalable-target \
    --service-namespace sagemaker \
    --resource-id inference-component/qwen3-asr-1.7b \
    --scalable-dimension sagemaker:inference-component:DesiredCopyCount \
    --min-capacity 1 \
    --max-capacity 8
```

**boto3:**
```python
aas = boto3.client("application-autoscaling", region_name="us-east-1")

aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="inference-component/qwen3-asr-1.7b",
    ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
    MinCapacity=1,
    MaxCapacity=8,
)
```

#### Attach a High-Resolution Scaling Policy

Now we can use the predefined `SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution` metric, which emits every 10 seconds:

```bash
aws application-autoscaling put-scaling-policy \
    --policy-name qwen3-asr-ic-scaling \
    --policy-type TargetTrackingScaling \
    --resource-id inference-component/qwen3-asr-1.7b \
    --service-namespace sagemaker \
    --scalable-dimension sagemaker:inference-component:DesiredCopyCount \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 1.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution"
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 60
    }'
```

**boto3:**
```python
aas.put_scaling_policy(
    PolicyName="qwen3-asr-ic-scaling",
    PolicyType="TargetTrackingScaling",
    ResourceId="inference-component/qwen3-asr-1.7b",
    ServiceNamespace="sagemaker",
    ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution",
        },
        "ScaleInCooldown": 600,
        "ScaleOutCooldown": 60,
    },
)
```

**Key differences from traditional scaling:**
- `ScaleOutCooldown: 60` (1 minute) — much shorter than the traditional 300s because new copies can often be placed on existing warm instances, avoiding the 8-15 minute cold start.
- `TargetValue: 1.0` — aim for 1 concurrent stream per copy. Since each copy occupies one GPU, this maintains the 1:1 stream-to-GPU ratio.
- SageMaker's `ManagedInstanceScaling` automatically provisions new instances when needed to host the additional copies.

#### How the Two-Level Scaling Works

```
                   Application Auto Scaling             SageMaker Managed
                   (you configure)                      (automatic)
                         │                                    │
New stream arrives       │                                    │
         ↓               │                                    │
ConcurrentRequests > 1   │                                    │
per copy                 │                                    │
         ↓               │                                    │
Scale out: CopyCount     │                                    │
1 → 2                    │                                    │
         ↓               │                                    │
                         │  No instance has room for          │
                         │  copy #2 (each needs 1 GPU)        │
                         │                                    │
                         │                    SageMaker adds  │
                         │                    Instance 2      │
                         │                         ↓          │
                         │                    Copy #2 placed  │
                         │                    on Instance 2   │
```

You only configure scaling for the inference component (copy count). SageMaker handles instance provisioning automatically.

### Container and Application Changes

**None required.** The container contract is the same:
- `GET /ping` → 200 (health check)
- `WS /invocations-bidirectional-stream` → streaming ASR
- Docker label `com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true`

SageMaker still downloads the model artifact to `/opt/ml/model/`, starts the container, and routes WebSocket traffic to port 8080. The only difference is in how the infrastructure is provisioned (endpoint config + inference component instead of model + endpoint config).

### Invocation Changes

For traditional endpoints, the client connects with just the endpoint name:

```python
stream = await client.invoke_endpoint_with_bidirectional_stream(
    InvokeEndpointWithBidirectionalStreamInput(endpoint_name="qwen3-asr-bidi-streaming")
)
```

With inference components, you can optionally target a specific component (useful when multiple models share an endpoint):

```python
stream = await client.invoke_endpoint_with_bidirectional_stream(
    InvokeEndpointWithBidirectionalStreamInput(
        endpoint_name="qwen3-asr-bidi-streaming",
        # Optional: target specific component when hosting multiple models
        # inference_component_name="qwen3-asr-1.7b",
    )
)
```

When there's only one inference component on the endpoint, the component name is optional — SageMaker routes to the only available component automatically.

### Cleanup of the Old Deployment

If migrating an existing traditional endpoint:

```bash
# 1. Delete the old endpoint
aws sagemaker delete-endpoint --endpoint-name qwen3-asr-bidi-streaming

# 2. Delete the old endpoint config
aws sagemaker delete-endpoint-config \
    --endpoint-config-name qwen3-asr-bidi-streaming-config

# 3. Delete the old model (not needed with inference components)
aws sagemaker delete-model --model-name qwen3-asr-bidi-streaming-model

# 4. Deploy the new inference component-based endpoint
# (follow steps 1-4 above)
```

Note: The ECR image and S3 model artifact are reused — only the SageMaker resources change.

### Migration Effort

| Task | Effort |
|------|--------|
| Rewrite `deploy.py` to use inference components | 1-2 hours |
| Configure auto-scaling with high-resolution metrics | 30 minutes |
| Test end-to-end (deploy, invoke, scale, cleanup) | 2-3 hours |
| Update `client.py` to optionally pass component name | 15 minutes |
| Update documentation (README.md, SCALING.md) | 1 hour |
| **Total** | **~half a day** |

No container changes, no model changes, no code changes in `app.py`.

---

## Connection Concurrency Guard

### The Problem

The current `app.py` accepts **all incoming WebSocket connections** without any limit. Since vLLM's synchronous `LLM.generate()` serializes inference internally, concurrent streams don't fail — they silently degrade. Each additional stream roughly doubles per-chunk latency for all active streams, with no feedback to the client.

What happens today with two concurrent connections:

```
Connection 1:  connects → start → audio → [inference].............[inference]
Connection 2:  connects → start → audio ..........→ [inference].............[inference]
                                                     ↑ blocked behind conn 1
```

Neither client receives an error. The second client's audio chunks queue up while waiting for inference, and responses arrive late. If latency gets bad enough, SageMaker's connection timeout or WebSocket ping/pong may kill the connection entirely.

### The Fix

Add an `asyncio.Semaphore` to `app.py` to limit active sessions to one at a time. When the semaphore is full, reject additional connections immediately with a clear error so SageMaker can route them to another instance (with horizontal scaling) or the client can retry.

#### Changes to `app.py`

```python
import asyncio

# Maximum concurrent streaming sessions per container.
# Set to 1 for the synchronous vLLM LLM backend (inference serializes).
# Increase if using AsyncLLMEngine with continuous batching.
MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "1"))

_session_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SESSIONS)
```

Update the WebSocket handler to acquire the semaphore before processing:

```python
@app.websocket("/invocations-bidirectional-stream")
async def websocket_invoke(websocket: WebSocket):
    await websocket.accept()

    # Reject if at capacity
    if not _session_semaphore.locked() or MAX_CONCURRENT_SESSIONS > 1:
        acquired = _session_semaphore.acquire(blocking=False)  # non-blocking
    else:
        acquired = False

    if not acquired:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Server at capacity. Try again or connect to another instance.",
        }))
        await websocket.close(code=1013)  # 1013 = Try Again Later
        logger.warning("Rejected connection: at capacity (%d/%d)",
                       MAX_CONCURRENT_SESSIONS, MAX_CONCURRENT_SESSIONS)
        return

    try:
        # ... existing session logic (start → audio → finish) ...
        pass
    finally:
        _session_semaphore.release()
```

A simpler and more idiomatic approach using `Semaphore` as a context manager with a non-blocking try:

```python
@app.websocket("/invocations-bidirectional-stream")
async def websocket_invoke(websocket: WebSocket):
    await websocket.accept()

    try:
        await asyncio.wait_for(_session_semaphore.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Server at capacity. Try again or connect to another instance.",
        }))
        await websocket.close(code=1013)
        logger.warning("Rejected connection: at capacity")
        return

    try:
        state = None
        sample_format = "float32"
        session_started = False

        # ... existing message loop ...
    finally:
        _session_semaphore.release()
```

#### WebSocket Close Code

[RFC 6455 close code 1013](https://www.rfc-editor.org/rfc/rfc6455#section-7.4.1) means "Try Again Later" — semantically correct for a capacity rejection. This signals to SageMaker's router that the instance is busy, which helps with connection routing when multiple instances are available.

#### Environment Variable

The `MAX_CONCURRENT_SESSIONS` environment variable (default `1`) allows tuning if the async engine rewrite is implemented later:

```dockerfile
# In Dockerfile — default to 1 for sync vLLM backend
ENV MAX_CONCURRENT_SESSIONS=1
```

For the async engine rewrite (Path B in [SCALING.md](SCALING.md#path-b-async-engine-rewrite)), increase this to allow multiple concurrent streams per GPU:

```dockerfile
ENV MAX_CONCURRENT_SESSIONS=3
```

### How This Interacts with Auto Scaling

The concurrency guard is critical for horizontal scaling to work correctly:

```
Without guard:                        With guard:

Client A → Instance 1 (serving A)    Client A → Instance 1 (serving A)
Client B → Instance 1 (degraded!)    Client B → Instance 1 → rejected (1013)
                                     Client B → Instance 2 (routed by SageMaker)
```

When SageMaker's router receives a connection rejection (close code 1013), it can route the request to another instance. Combined with auto scaling on `ConcurrentRequestsPerModel`, the system:

1. Rejects the overflow connection
2. SageMaker routes it to another available instance (if one exists)
3. CloudWatch metric reflects the spike in concurrent requests
4. Auto scaling triggers scale-out for additional capacity

Without the guard, SageMaker has no signal that the instance is overloaded — the connection succeeds, and latency silently degrades.

### Effort

| Task | Effort |
|------|--------|
| Add semaphore and rejection logic to `app.py` | 30 minutes |
| Add `MAX_CONCURRENT_SESSIONS` env var to Dockerfile | 5 minutes |
| Update `test_local.py` with concurrent connection test | 30 minutes |
| Test on deployed endpoint | 1 hour |
| **Total** | **~2 hours** |

---

## Other Potential Enhancements

### Async vLLM Engine for Single-GPU Concurrency

Replace vLLM's synchronous `LLM` class with `AsyncLLMEngine` to enable true concurrent inference via continuous batching on a single GPU. See [SCALING.md, Path B](SCALING.md#path-b-async-engine-rewrite) for full details.

**Effort**: ~3-5 days (upstream library changes required)

### Multi-Language Endpoint with Separate Components

With inference components, you could deploy language-specific fine-tuned models on the same endpoint:

```
Endpoint: qwen3-asr-multilingual
  ├── InferenceComponent: qwen3-asr-english  (optimized for English)
  ├── InferenceComponent: qwen3-asr-chinese  (optimized for Chinese)
  └── InferenceComponent: qwen3-asr-general  (auto-detect, fallback)
```

Clients would route to the appropriate component based on known language context, reducing inference cost for known-language streams.

### Warm Pool for Faster Scale-Out

SageMaker supports [warm pools](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-warm-pools.html) that keep instances in a pre-initialized state after scale-in. This reduces the cold start from 8-15 minutes to 2-3 minutes for scale-out events. Enable by setting `RoutingConfig` in the endpoint config:

```python
sm.create_endpoint_config(
    EndpointConfigName="qwen3-asr-ic-config",
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "InstanceType": "ml.g6.xlarge",
        "InitialInstanceCount": 1,
        "ManagedInstanceScaling": {
            "Status": "ENABLED",
            "MinInstanceCount": 1,
            "MaxInstanceCount": 4,
        },
        "RoutingConfig": {
            "RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS",
        },
    }],
)
```

Note: Warm pools incur charges for the idle instances at a reduced rate.

### Streaming CloudWatch Metrics Dashboard

Create a CloudWatch dashboard to monitor streaming performance in real-time:

```bash
aws cloudwatch put-dashboard --dashboard-name qwen3-asr-streaming \
    --dashboard-body '{
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/SageMaker", "ConcurrentRequestsPerModel",
                         "EndpointName", "qwen3-asr-bidi-streaming",
                         "VariantName", "AllTraffic"],
                        ["AWS/SageMaker", "ModelLatency",
                         "EndpointName", "qwen3-asr-bidi-streaming",
                         "VariantName", "AllTraffic"],
                        ["AWS/SageMaker", "FirstChunkLatency",
                         "EndpointName", "qwen3-asr-bidi-streaming",
                         "VariantName", "AllTraffic"]
                    ],
                    "period": 60,
                    "title": "Streaming ASR Metrics"
                }
            }
        ]
    }'
```

Key metrics for streaming ASR:
- `ConcurrentRequestsPerModel` — active streaming connections
- `ModelLatency` — per-chunk inference time
- `FirstChunkLatency` — time to first partial transcription (user-perceived latency)
- `MidStreamErrors` — errors during active streaming sessions

### Token-Level Streaming to Client

Currently, the server waits for vLLM to complete the full `generate()` call for each audio chunk before sending a transcription update. With the async engine rewrite, it's possible to stream individual tokens to the client as they're generated:

```python
# Hypothetical token-level streaming:
async for output in engine.generate(inp, sampling_params, request_id):
    partial_text = output.outputs[0].text
    await websocket.send_text(json.dumps({
        "type": "transcription",
        "text": partial_text,
        "is_partial": True,  # Client knows more tokens are coming
    }))
```

This would reduce the perceived latency for each audio chunk from "wait for full generation" to "see first token immediately." The benefit is most noticeable on longer accumulated audio where generation takes more time.

**Effort**: Requires the async engine rewrite (Path B in SCALING.md) as a prerequisite.
