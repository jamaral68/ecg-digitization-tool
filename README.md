# ecg-digitization-tool

Projeto para digitalização de eletrocardiogramas (ECG) usando Python e técnicas de visão computacional / machine learning.

---

## Estrutura do projeto

```
ecg-digitization-tool/
├── src/                      # Código-fonte
├── notebooks/                # Notebooks de experimentação
├── datasets/                 # Imagens de ECG (não versionadas)
├── deploy/                   # Dockerfile + scripts para SageMaker (CUDA)
├── docs/                     # Documentação adicional
├── pyproject.toml            # Dependências para dev local (PyTorch ROCm)
├── Makefile                  # Atalhos para build/run/deploy
└── README.md
```

---

## Setup local

### 1. Pré-requisitos
- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- AWS CLI v2 (apenas se for usar o bucket S3)

### 2. Instalar dependências

```bash
poetry install
```

> O `pyproject.toml` da raiz usa **PyTorch ROCm** (AMD GPU). Para builds em CUDA (SageMaker), veja `deploy/`.

### 3. Acesso ao bucket S3

Para acessar o bucket AWS deste projeto, instale o [AWS CLI](https://docs.aws.amazon.com/pt_br/cli/latest/userguide/getting-started-install.html) e autentique:

```bash
aws configure
```

---

## Docker — execução local e deploy AWS

A imagem Docker (CUDA) é usada tanto para teste local quanto para deploy no SageMaker. Comandos via `make`:

| Comando | Descrição |
|---|---|
| `make build` | Build local da imagem (CUDA) |
| `make run` | Sobe Jupyter Lab em `http://localhost:8888` |
| `make shell` | Abre bash dentro do container |
| `make deploy` | Build + push para o Amazon ECR |
| `make register` | Registra a imagem no SageMaker Studio |
| `make help` | Lista todos os targets |

Detalhes da imagem, scripts e configurações: **[deploy/README.md](deploy/README.md)**.

---

## SageMaker Studio com imagem customizada

Tutorial completo para empacotar o projeto, publicar no ECR e usar como kernel no SageMaker Studio:

📘 **[docs/sagemaker-custom-image.md](docs/sagemaker-custom-image.md)**

Cobre: criação da IAM Role, registro da imagem, AppImageConfig, anexação ao Domain, troubleshooting e custos.

---

## Documentação adicional

- [Deploy e imagem Docker](deploy/README.md)
- [Tutorial SageMaker](docs/sagemaker-custom-image.md)
