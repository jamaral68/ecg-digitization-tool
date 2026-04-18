#!/usr/bin/env bash
set -euo pipefail

# Build da imagem Docker e push para o Amazon ECR.
# Uso:
#   ./deploy/scripts/build_and_push.sh [tag]
# Se a tag nao for passada, usa "latest".

AWS_REGION="${AWS_REGION:-sa-east-1}"
REPO_NAME="${REPO_NAME:-ecg-digitization-tool}"
TAG="${1:-latest}"

AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

# Garante que o repositorio existe (idempotente).
aws ecr describe-repositories \
    --repository-names "${REPO_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "${REPO_NAME}" \
       --region "${AWS_REGION}"

# Login do Docker no ECR.
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin \
      "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build com contexto na raiz do projeto (Dockerfile em deploy/).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build \
  --provenance=false \
  --sbom=false \
  -f "${PROJECT_ROOT}/deploy/Dockerfile" \
  -t "${REPO_NAME}:${TAG}" \
  "${PROJECT_ROOT}"

docker tag "${REPO_NAME}:${TAG}" "${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

echo ""
echo "Push concluido: ${ECR_URI}:${TAG}"
