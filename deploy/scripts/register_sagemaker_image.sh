#!/usr/bin/env bash
set -euo pipefail

# Registra a imagem do ECR como uma Custom Image do SageMaker Studio.
# Pre-requisitos:
#   - Imagem ja publicada no ECR (rode build_and_push.sh antes).
#   - IAM Role com AmazonSageMakerFullAccess criada.
#
# Uso:
#   SAGEMAKER_ROLE_ARN=arn:aws:iam::...:role/SageMakerExecutionRole \
#     ./deploy/scripts/register_sagemaker_image.sh [tag]

AWS_REGION="${AWS_REGION:-sa-east-1}"
REPO_NAME="${REPO_NAME:-ecg-digitization-tool}"
IMAGE_NAME="${IMAGE_NAME:-ecg-digitization-tool}"
APP_CONFIG_NAME="${APP_CONFIG_NAME:-ecg-digitization-tool-config}"
TAG="${1:-latest}"

: "${SAGEMAKER_ROLE_ARN:?Defina SAGEMAKER_ROLE_ARN com o ARN da IAM Role do SageMaker}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:${TAG}"

# Cria a SageMaker Image (idempotente).
aws sagemaker describe-image \
    --image-name "${IMAGE_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws sagemaker create-image \
       --image-name "${IMAGE_NAME}" \
       --role-arn "${SAGEMAKER_ROLE_ARN}" \
       --region "${AWS_REGION}"

# Nova versao apontando para a tag atual do ECR.
aws sagemaker create-image-version \
  --image-name "${IMAGE_NAME}" \
  --base-image "${ECR_URI}" \
  --region "${AWS_REGION}"

# Cria o AppImageConfig (idempotente).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_CONFIG_FILE="${SCRIPT_DIR}/../sagemaker/app-config.json"

aws sagemaker describe-app-image-config \
    --app-image-config-name "${APP_CONFIG_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws sagemaker create-app-image-config \
       --cli-input-json "file://${APP_CONFIG_FILE}" \
       --region "${AWS_REGION}"

echo ""
echo "Imagem registrada. Anexe '${IMAGE_NAME}' ao seu Domain do Studio pelo console."
