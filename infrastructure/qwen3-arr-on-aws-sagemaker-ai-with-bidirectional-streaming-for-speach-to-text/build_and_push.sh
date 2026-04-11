#!/usr/bin/env bash
# Build the Qwen3-ASR SageMaker bidirectional streaming container and push to ECR.
#
# Usage:
#   ./build_and_push.sh                          # defaults
#   ./build_and_push.sh my-repo-name latest      # custom repo name / tag
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker installed
#   - Permissions: ecr:CreateRepository, ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability,
#     ecr:PutImage, ecr:InitiateLayerUpload, ecr:UploadLayerPart, ecr:CompleteLayerUpload

set -euo pipefail

CONTAINER_NAME="${1:-qwen3-asr-sagemaker}"
CONTAINER_TAG="${2:-latest}"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REGION=${REGION:-us-east-1}

IMAGE_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${CONTAINER_NAME}:${CONTAINER_TAG}"

echo "============================================"
echo "  Container:  ${CONTAINER_NAME}:${CONTAINER_TAG}"
echo "  Account:    ${ACCOUNT}"
echo "  Region:     ${REGION}"
echo "  Image URI:  ${IMAGE_URI}"
echo "============================================"

# Create ECR repository if it doesn't exist
if ! aws ecr describe-repositories --repository-names "${CONTAINER_NAME}" --region "${REGION}" > /dev/null 2>&1; then
    echo "Creating ECR repository: ${CONTAINER_NAME}"
    aws ecr create-repository --repository-name "${CONTAINER_NAME}" --region "${REGION}" > /dev/null
fi

# Authenticate Docker with ECR
aws ecr get-login-password --region "${REGION}" \
    | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Build
echo "Building Docker image..."
docker build --platform linux/amd64 --provenance=false -t "${CONTAINER_NAME}" .

# Tag and push
docker tag "${CONTAINER_NAME}" "${IMAGE_URI}"
echo "Pushing to ECR..."
docker push "${IMAGE_URI}"

echo ""
echo "Done. Image URI:"
echo "  ${IMAGE_URI}"
