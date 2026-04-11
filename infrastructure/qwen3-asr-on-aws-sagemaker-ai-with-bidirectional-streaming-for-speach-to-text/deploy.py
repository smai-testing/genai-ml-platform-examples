#!/usr/bin/env python3
"""
Create a SageMaker real-time endpoint for Qwen3-ASR with bidirectional streaming.

Usage:
    python deploy.py \
        --image-uri 123456789012.dkr.ecr.us-west-2.amazonaws.com/qwen3-asr-sagemaker:latest \
        --model-data-url s3://my-bucket/qwen3-asr/model.tar.gz \
        --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
        --region us-west-2
"""

import argparse
import time

import boto3


def parse_args():
    p = argparse.ArgumentParser(description="Deploy Qwen3-ASR to SageMaker")
    p.add_argument("--image-uri", required=True, help="ECR image URI for the container")
    p.add_argument("--model-data-url", required=True,
                   help="S3 URI of model.tar.gz (e.g. s3://bucket/qwen3-asr/model.tar.gz)")
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--region", default="us-west-2", help="AWS region")
    p.add_argument("--instance-type", default="ml.g5.xlarge",
                   help="SageMaker instance type (default: ml.g5.xlarge)")
    p.add_argument("--endpoint-name", default="qwen3-asr-bidi-streaming",
                   help="Endpoint name")
    p.add_argument("--wait", action="store_true", help="Wait for the endpoint to be InService")
    return p.parse_args()


def main():
    args = parse_args()
    sm = boto3.client("sagemaker", region_name=args.region)

    model_name = f"{args.endpoint_name}-model"
    config_name = f"{args.endpoint_name}-config"
    endpoint_name = args.endpoint_name

    # 1. Create model
    print(f"Creating model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": args.image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": args.model_data_url,
        },
        ExecutionRoleArn=args.role,
    )

    # 2. Create endpoint config
    print(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": args.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )

    # 3. Create endpoint
    print(f"Creating endpoint: {endpoint_name}")
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    print(f"Endpoint '{endpoint_name}' creation initiated.")

    if args.wait:
        print("Waiting for endpoint to be InService...")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"Delay": 30, "MaxAttempts": 60},
        )
        print(f"Endpoint '{endpoint_name}' is now InService.")
    else:
        print("Use --wait to block until the endpoint is ready, or check status with:")
        print(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name} --query EndpointStatus")


if __name__ == "__main__":
    main()
