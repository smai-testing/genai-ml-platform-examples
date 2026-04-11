#!/usr/bin/env python3
"""
Example client for invoking a Qwen3-ASR SageMaker endpoint with bidirectional streaming.

Uses the experimental aws-sdk-sagemaker-runtime-http2 SDK to send audio and receive
partial transcriptions in real-time.

Install the SDK:
    pip install aws-sdk-sagemaker-runtime-http2 librosa boto3

Usage:
    python client.py --endpoint-name qwen3-asr-bidi-streaming --audio-file test.wav
    python client.py --endpoint-name qwen3-asr-bidi-streaming --audio-file test.wav --language Chinese
"""

import argparse
import asyncio
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen3-asr-client")

AWS_REGION = "us-east-1"


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-ASR bidirectional streaming client")
    p.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    p.add_argument("--audio-file", required=True, help="Path to audio file (WAV, MP3, etc.)")
    p.add_argument("--region", default=AWS_REGION, help="AWS region")
    p.add_argument("--language", default=None, help="Force language (e.g. English, Chinese)")
    p.add_argument("--chunk-duration-ms", type=int, default=500,
                   help="Duration of each audio chunk sent (milliseconds)")
    p.add_argument("--chunk-size-sec", type=float, default=2.0,
                   help="Server-side ASR chunk size in seconds")
    p.add_argument("--profile", default=None,
                   help="AWS profile name (uses default credential chain if not set)")
    return p.parse_args()


def load_audio_as_pcm_float32(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and return 16 kHz mono float32 PCM."""
    # import librosa
    # wav, _ = librosa.load(path, sr=target_sr, mono=True)
    import soundfile as sf
    from scipy.signal import resample

    wav, sr = sf.read(path, dtype='float32')
    if wav.ndim > 1:  # stereo to mono
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = resample(wav, int(len(wav) * target_sr / sr)).astype(np.float32)
    return wav.astype(np.float32)

def resolve_credentials(region: str, profile: str | None = None):
    """
    Resolve AWS credentials using boto3's standard credential chain and set them
    as environment variables for the smithy EnvironmentCredentialsResolver.

    This handles AWS profiles, SSO, IAM roles, env vars, etc.
    """
    import boto3

    session = boto3.Session(region_name=region, profile_name=profile)
    credentials = session.get_credentials()

    if credentials is None:
        raise RuntimeError(
            "AWS credentials not found. Configure via:\n"
            "  1. AWS CLI: aws configure\n"
            "  2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
            "  3. AWS credentials file: ~/.aws/credentials\n"
            "  4. IAM role (when running on AWS infrastructure)"
        )

    frozen = credentials.get_frozen_credentials()
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token

    # Log identity for debugging
    sts = session.client("sts")
    identity = sts.get_caller_identity()
    logger.info("Authenticated as: %s", identity.get("Arn", "Unknown"))


async def run(args):
    try:
        from aws_sdk_sagemaker_runtime_http2.client import SageMakerRuntimeHTTP2Client
        from aws_sdk_sagemaker_runtime_http2.config import Config, HTTPAuthSchemeResolver
        from aws_sdk_sagemaker_runtime_http2.models import (
            InvokeEndpointWithBidirectionalStreamInput,
            RequestPayloadPart,
            RequestStreamEventPayloadPart,
        )
        from smithy_aws_core.auth.sigv4 import SigV4AuthScheme
        from smithy_aws_core.identity import EnvironmentCredentialsResolver
    except ImportError:
        logger.error(
            "The experimental SDK 'aws-sdk-sagemaker-runtime-http2' is required.\n"
            "Install it with: pip install aws-sdk-sagemaker-runtime-http2\n"
            "See: https://aws.amazon.com/blogs/machine-learning/"
            "introducing-bidirectional-streaming-for-real-time-inference-on-amazon-sagemaker-ai/"
        )
        sys.exit(1)

    # Resolve credentials via boto3 (handles profiles, SSO, IAM roles, etc.)
    resolve_credentials(args.region, args.profile)

    # Load audio
    logger.info("Loading audio: %s", args.audio_file)
    pcm = load_audio_as_pcm_float32(args.audio_file)
    logger.info("Audio loaded: %.2f seconds (%d samples)", len(pcm) / 16000, len(pcm))

    # Initialize client
    bidi_endpoint = f"https://runtime.sagemaker.{args.region}.amazonaws.com:8443"
    config = Config(
        endpoint_uri=bidi_endpoint,
        region=args.region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        auth_scheme_resolver=HTTPAuthSchemeResolver(),
        auth_schemes={"aws.auth#sigv4": SigV4AuthScheme(service="sagemaker")},
    )
    client = SageMakerRuntimeHTTP2Client(config=config)

    # Start bidirectional stream
    logger.info("Connecting to endpoint: %s", args.endpoint_name)
    stream = await client.invoke_endpoint_with_bidirectional_stream(
        InvokeEndpointWithBidirectionalStreamInput(endpoint_name=args.endpoint_name)
    )

    # Get output stream
    output = await stream.await_output()
    output_stream = output[1]

    # Task to process responses
    async def process_responses():
        try:
            while True:
                result = await output_stream.receive()
                if result is None:
                    break
                if result.value and result.value.bytes_:
                    data = json.loads(result.value.bytes_.decode("utf-8"))
                    msg_type = data.get("type", "")
                    if msg_type == "transcription":
                        logger.info("[partial] lang=%s text=%s", data.get("language"), data.get("text"))
                    elif msg_type == "final":
                        logger.info("[final]   lang=%s text=%s", data.get("language"), data.get("text"))
                    elif msg_type == "error":
                        logger.error("[error]   %s", data.get("message"))
                    else:
                        logger.info("[unknown] %s", data)
        except Exception as e:
            logger.error("Response processing error: %s", e)

    response_task = asyncio.create_task(process_responses())

    # Send start message
    start_msg = json.dumps({
        "type": "start",
        "language": args.language,
        "context": "",
        "chunk_size_sec": args.chunk_size_sec,
        "sample_format": "float32",
    }).encode("utf-8")

    await stream.input_stream.send(
        RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=start_msg))
    )
    logger.info("Session started")

    # Stream audio in chunks
    chunk_samples = int(16000 * args.chunk_duration_ms / 1000)
    offset = 0
    chunk_num = 0
    while offset < len(pcm):
        chunk = pcm[offset : offset + chunk_samples]
        audio_bytes = chunk.tobytes()

        await stream.input_stream.send(
            RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=audio_bytes))
        )
        offset += chunk_samples
        chunk_num += 1
        logger.info("Sent chunk %d (%d bytes)", chunk_num, len(audio_bytes))

        # Simulate real-time pace
        await asyncio.sleep(args.chunk_duration_ms / 1000.0)

    # Send finish message
    finish_msg = json.dumps({"type": "finish"}).encode("utf-8")
    await stream.input_stream.send(
        RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=finish_msg))
    )
    logger.info("Finish message sent, waiting for final response...")

    # Wait for responses to complete, then close
    await asyncio.sleep(10)
    await stream.input_stream.close()

    if not response_task.done():
        try:
            await asyncio.wait_for(response_task, timeout=5.0)
        except asyncio.TimeoutError:
            response_task.cancel()

    logger.info("Done.")


def main():
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
