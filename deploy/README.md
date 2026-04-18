# Deploy — Imagem Docker para SageMaker

Configuração de ambiente para rodar o projeto no Amazon SageMaker.

## Estrutura

```
deploy/
├── Dockerfile                    # Imagem com dependências CUDA
├── pyproject.sagemaker.toml      # Versão do pyproject com torch CUDA (não ROCm)
├── .dockerignore
├── scripts/
│   ├── build_and_push.sh         # Build + push para ECR
│   └── register_sagemaker_image.sh  # Registra imagem no SageMaker Studio
└── sagemaker/
    └── app-config.json           # Config do kernel para Studio
```

## Por que um `pyproject.sagemaker.toml` separado?

O `pyproject.toml` da raiz usa PyTorch **ROCm** (AMD GPU). SageMaker roda em **NVIDIA GPU**, então precisamos de builds CUDA. Mantemos os dois arquivos para não quebrar o dev local com AMD.

## Uso

### 1. Build e push para o ECR

```bash
# Configure AWS CLI antes: aws configure
./deploy/scripts/build_and_push.sh              # tag "latest"
./deploy/scripts/build_and_push.sh v1           # tag versionada
```

Variáveis opcionais:
- `AWS_REGION` (default: `sa-east-1`)
- `REPO_NAME` (default: `ecg-digitization-tool`)

### 2. Registrar a imagem no SageMaker Studio

```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/SageMakerExecutionRole
./deploy/scripts/register_sagemaker_image.sh
```

Depois, no console do SageMaker: **Domains → seu domínio → Environment → Attach image**.

### 3. Teste local (opcional)

```bash
docker build -f deploy/Dockerfile -t ecg-digitization-tool:dev .
docker run --rm -p 8888:8888 ecg-digitization-tool:dev
# Acesse http://localhost:8888
```

## Manutenção

Quando adicionar/atualizar dependências no `pyproject.toml` da raiz, **replique no `deploy/pyproject.sagemaker.toml`** (exceto `torch`/`torchvision`, que ficam CUDA aqui).
