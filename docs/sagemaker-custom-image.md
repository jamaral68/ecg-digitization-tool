# Tutorial — Notebook no SageMaker com imagem customizada

Guia completo para empacotar o projeto numa imagem Docker, publicar no Amazon ECR e usar no SageMaker Studio como kernel customizado.

---

## Visão geral

Para usar uma imagem própria no Studio, a AWS precisa de **4 objetos** conectados:

```
ECR (Elastic Container Registry)
  └── ecg-digitization-tool:latest          ← imagem Docker
         ▲
         │ referenciada por
SageMaker Image
  └── ecg-digitization-tool                 ← apelido no SageMaker
         └── ImageVersion 1                 ← cada push gera uma nova versão
                ▲
                │ usada por
AppImageConfig
  └── ecg-digitization-tool-config          ← diz ao Studio como rodar (kernel, uid/gid)
         ▲
         │ anexada ao
SageMaker Domain
  └── seu-dominio-studio                    ← onde o kernel aparece no Studio
```

Fluxo: **ECR → SageMaker Image → AppImageConfig → Domain → Studio**.

---

## Pré-requisitos

- AWS CLI v2 instalado e configurado (`aws configure`)
- Docker instalado
- Acesso à conta AWS com permissão para criar IAM Roles, repositórios ECR, e recursos SageMaker
- Imagem já buildada e publicada no ECR (rodar `make deploy` antes — ver [README do deploy](../deploy/README.md))

---

## Passo 1 — Criar a IAM Role de execução

O Studio precisa de uma role que permita ao serviço puxar a imagem do ECR e acessar S3, CloudWatch etc.

### Pelo console (recomendado)

1. Console → **IAM → Roles → Create role**.
2. **Trusted entity type**: `AWS service`.
3. **Use case**: selecione `SageMaker` → `SageMaker - Execution`.
4. Next → mantém `AmazonSageMakerFullAccess` (já vem marcada).
5. **Role name**: `SageMakerExecutionRole-ECG` → Create role.
6. Volte na role criada → aba **Permissions** → **Add permissions → Attach policies**.
7. Marque (na mesma tela):
   - `AmazonEC2ContainerRegistryReadOnly` — para puxar do ECR
   - `AmazonS3FullAccess` (ou `AmazonS3ReadOnlyAccess` se só vai ler)
8. Add permissions.
9. Copie o **ARN** da role — formato:
   ```
   arn:aws:iam::123456789012:role/SageMakerExecutionRole-ECG
   ```

### Pela CLI

```bash
cat > trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name SageMakerExecutionRole-ECG \
  --assume-role-policy-document file://trust.json

for POLICY in \
  arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
  arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly \
  arn:aws:iam::aws:policy/AmazonS3FullAccess; do
  aws iam attach-role-policy \
    --role-name SageMakerExecutionRole-ECG \
    --policy-arn $POLICY
done
```

> **Erro `iam:CreateRole` not authorized?** Seu usuário IAM não tem permissão para criar roles. Opções:
> - Se você é admin da conta: anexe `IAMFullAccess` ao seu usuário.
> - Se é conta corporativa: peça ao admin para criar a role e te conceder `iam:PassRole` sobre ela.

---

## Passo 2 — Registrar a imagem no SageMaker

Cria o recurso **SageMaker Image** que aponta para sua imagem no ECR.

```bash
export AWS_REGION=sa-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ROLE_ARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/SageMakerExecutionRole-ECG

aws sagemaker create-image \
  --image-name ecg-digitization-tool \
  --role-arn $ROLE_ARN \
  --region $AWS_REGION
```

| Campo | Função |
|---|---|
| `--image-name` | Apelido interno da imagem no SageMaker. |
| `--role-arn` | Role que o SageMaker assume para puxar do ECR. |

Verificação:
```bash
aws sagemaker describe-image --image-name ecg-digitization-tool --region $AWS_REGION
```

---

## Passo 3 — Criar a primeira versão da imagem

Toda SageMaker Image é versionada. Cada push novo no ECR → nova `ImageVersion`.

```bash
aws sagemaker create-image-version \
  --image-name ecg-digitization-tool \
  --base-image $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ecg-digitization-tool:latest \
  --region $AWS_REGION
```

**Pontos importantes**:
- O SageMaker congela o **digest** da imagem no momento desse comando — sobrescrever a tag no ECR depois não afeta versões já criadas.
- A criação demora 30-60s validando a imagem. Acompanhe com:
  ```bash
  aws sagemaker describe-image-version \
    --image-name ecg-digitization-tool \
    --region $AWS_REGION
  ```
  Aguarde `ImageVersionStatus: CREATED`. Se der `CREATE_FAILED`, leia `FailureReason`.

---

## Passo 4 — Criar o AppImageConfig

Diz ao Studio **como rodar a imagem**: nome do kernel, mount point, uid/gid.

O arquivo já existe em `deploy/sagemaker/app-config.json`:

```json
{
  "AppImageConfigName": "ecg-digitization-tool-config",
  "KernelGatewayImageConfig": {
    "KernelSpecs": [
      { "Name": "ecg-digitization-tool", "DisplayName": "Python (ECG)" }
    ],
    "FileSystemConfig": {
      "MountPath": "/home/sagemaker-user",
      "DefaultUid": 1000,
      "DefaultGid": 100
    }
  }
}
```

| Campo | Significado |
|---|---|
| `AppImageConfigName` | Nome do config (selecionado ao anexar a imagem). |
| `KernelSpecs.Name` | **Deve bater com o nome do kernel instalado no Dockerfile** (`python -m ipykernel install --name ...`). |
| `KernelSpecs.DisplayName` | Nome exibido no menu "Select Kernel" do Studio. |
| `MountPath` | Onde o EFS pessoal do usuário Studio é montado. |
| `DefaultUid`/`DefaultGid` | `1000/100` — padrão SageMaker. |

> ⚠️ Se mudar o `KernelSpecs.Name`, ajuste também o `--name` no `python -m ipykernel install` do `deploy/Dockerfile` para manter consistência.

Aplicar:

```bash
aws sagemaker create-app-image-config \
  --cli-input-json file://deploy/sagemaker/app-config.json \
  --region $AWS_REGION
```

---

## Passo 5 — Anexar ao Studio Domain

### Se ainda não tem um Domain

1. Console → **SageMaker → Domains → Create domain** → `Quick setup`.
2. **Domain name**: `ecg-domain`.
3. **Execution role**: `SageMakerExecutionRole-ECG`.
4. **VPC**: sua VPC default.
5. Submit (~5 min).

### Anexar a imagem ao Domain

1. Console → **SageMaker → Domains → ecg-domain → Environment**.
2. **Custom images for personal Studio apps → Attach image**.
3. **Existing image**:
   - Image: `ecg-digitization-tool`
   - Version: `1` (ou a mais recente)
4. Next → **AppImageConfig**: `ecg-digitization-tool-config`.
5. **Image type**: `JupyterLab image` (ou `KernelGateway image`).
6. Attach image.

### Via CLI (alternativa)

```bash
aws sagemaker update-domain \
  --domain-id <SEU_DOMAIN_ID> \
  --default-user-settings '{
    "KernelGatewayAppSettings": {
      "CustomImages": [{
        "ImageName": "ecg-digitization-tool",
        "ImageVersionNumber": 1,
        "AppImageConfigName": "ecg-digitization-tool-config"
      }]
    }
  }' \
  --region $AWS_REGION
```

---

## Passo 6 — Usar no Studio

1. Console → **SageMaker → Studio → Open Studio**.
2. **File → New → Notebook**.
3. **Select Kernel**: `Python (ECG)`.
4. **Instance type**: `ml.t3.medium` (CPU teste) ou `ml.g4dn.xlarge` (GPU).
5. Select → primeira inicialização demora 2-5 min puxando do ECR.

### Validação

```python
import sys, torch, cv2, pytesseract
print("Python:", sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Tesseract:", pytesseract.get_tesseract_version())
```

---

## Atualizando a imagem depois

Quando mudar código/dependências:

```bash
make deploy                                  # novo push no ECR

aws sagemaker create-image-version \         # nova versão SageMaker
  --image-name ecg-digitization-tool \
  --base-image $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/ecg-digitization-tool:latest \
  --region $AWS_REGION
```

No Studio: **pare e reinicie o app** para pegar a nova versão. Se fixou versão específica no Domain, atualize com `update-domain`.

---

## Erros comuns

| Erro | Causa provável |
|---|---|
| `create-image-version` falha com `CREATE_FAILED` | Role sem `AmazonEC2ContainerRegistryReadOnly`, ou imagem com manifest list incompatível (resolvido com `--provenance=false` no build). |
| Kernel `Python (ECG)` não aparece no Studio | `KernelSpecs.Name` no AppImageConfig ≠ `--name` do `ipykernel install` no Dockerfile. |
| `Cannot pull image` | Execution role sem permissão no ECR, ou ECR em região diferente do Domain. |
| `iam:CreateRole not authorized` | Usuário IAM não tem permissão para criar roles — peça ao admin. |
| Studio abre mas kernel crasha | Falta `jupyter`/`ipykernel` na imagem (já incluídos no `deploy/Dockerfile`). |

---

## Custos a observar

- **ECR**: ~$0.10/GB/mês de storage. Limpe versões antigas.
- **SageMaker Studio app**: cobra por hora de uso da instância. **Pare quando não estiver usando**:
  ```bash
  # Lista apps em execução
  aws sagemaker list-apps --region $AWS_REGION
  # Para
  aws sagemaker delete-app --domain-id <DOMAIN_ID> \
    --user-profile-name <USER> --app-type KernelGateway --app-name <APP_NAME>
  ```
