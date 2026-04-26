# ============================================================================
# Variaveis
# ============================================================================

AWS_REGION ?= sa-east-1
REPO_NAME  ?= ecg-digitization-tool
TAG        ?= latest
IMAGE      ?= $(REPO_NAME):$(TAG)
PORT       ?= 8888
STREAMLIT_PORT ?= 8501

PYTHON := python3
VENV_PATH := $(shell poetry env info --path 2>/dev/null)
PIP := $(VENV_PATH)/bin/pip
 
# Torch ROCm config
TORCH_VERSION := 2.11.0+rocm7.2
TORCH_INDEX := https://download.pytorch.org/whl/rocm7.2
 
.PHONY: help build run shell push deploy register deploy-full clean install-poetry app demo install install-torch install-amd-torch test-gpu help clean

help:
	@echo "Targets disponiveis:"
	@echo ""
	@echo "  Setup:"
	@echo "    install-poetry - Instala Poetry e dependencias do projeto"
	@echo ""
	@echo "  Streamlit:"
	@echo "    app           - Roda o app Streamlit localmente (porta $(STREAMLIT_PORT))"
	@echo "    demo          - Roda o app + tunel ngrok para demos"
	@echo ""
	@echo "  Docker / SageMaker:"
	@echo "    build         - Build local da imagem SageMaker (CUDA)"
	@echo "    run           - Roda o container local com Jupyter em :$(PORT)"
	@echo "    shell         - Abre bash dentro do container para debug"
	@echo "    push          - Build + push para o ECR"
	@echo "    deploy        - Alias para push (fluxo padrao)"
	@echo "    register      - Registra a imagem no SageMaker Studio"
	@echo "    deploy-full   - deploy + register"
	@echo "    clean         - Remove a imagem local"
	@echo ""
	@echo "  Variaveis (sobrescrever com VAR=valor):"
	@echo "    AWS_REGION=$(AWS_REGION)"
	@echo "    REPO_NAME=$(REPO_NAME)"
	@echo "    TAG=$(TAG)"
	@echo "    PORT=$(PORT)"

build:
	docker build \
	  --provenance=false \
	  --sbom=false \
	  -f deploy/Dockerfile -t $(IMAGE) .

run:
	docker run --rm -it \
	  -p $(PORT):8888 \
	  -u $$(id -u):$$(id -g) \
	  -v $(CURDIR)/src:/opt/ml/code/src \
	  -v $(CURDIR)/notebooks:/opt/ml/code/notebooks \
	  --name ecg-digitization-local \
	  $(IMAGE)

shell:
	docker run --rm -it \
	  -u $$(id -u):$$(id -g) \
	  -v $(CURDIR)/src:/opt/ml/code/src \
	  -v $(CURDIR)/notebooks:/opt/ml/code/notebooks \
	  $(IMAGE) bash

push:
	AWS_REGION=$(AWS_REGION) REPO_NAME=$(REPO_NAME) \
	  ./deploy/scripts/build_and_push.sh $(TAG)

deploy: push

register:
	@if [ -z "$(SAGEMAKER_ROLE_ARN)" ]; then \
	  echo "ERRO: defina SAGEMAKER_ROLE_ARN"; \
	  echo "  ex: make register SAGEMAKER_ROLE_ARN=arn:aws:iam::123:role/SageMakerExecutionRole"; \
	  exit 1; \
	fi
	AWS_REGION=$(AWS_REGION) REPO_NAME=$(REPO_NAME) \
	SAGEMAKER_ROLE_ARN=$(SAGEMAKER_ROLE_ARN) \
	  ./deploy/scripts/register_sagemaker_image.sh $(TAG)

deploy-full: deploy register

clean:
	-docker rmi $(IMAGE) 2>/dev/null || true


app:
	poetry run streamlit run app/app.py --server.port $(STREAMLIT_PORT)

demo:
	poetry run python app/serve_ngrok.py --port $(STREAMLIT_PORT)

install-poetry:
	@command -v poetry >/dev/null 2>&1 || curl -sSL https://install.python-poetry.org | python3 -
	@PATH="$$HOME/.local/bin:$$PATH" poetry install
	@echo ""
	@echo ">> Para usar 'poetry' no terminal, adicione ao seu ~/.bashrc:"
	@echo '     export PATH="$$HOME/.local/bin:$$PATH"'



install:
	@echo ">>> Instalando dependências do projeto via Poetry..."
	poetry install
	@echo ">>> Pronto! Use 'make install-amd-torch' para instalar o PyTorch com ROCm."
 
install-amd-torch:
	@echo ">>> Instalando PyTorch $(TORCH_VERSION) com suporte ROCm..."
	@if [ -z "$(VENV_PATH)" ]; then \
		echo "ERRO: Nenhum ambiente virtual Poetry encontrado. Rode 'make install' primeiro."; \
		exit 1; \
	fi
	$(PIP) install torch==$(TORCH_VERSION) torchvision \
		--index-url $(TORCH_INDEX)
	@echo ">>> PyTorch com ROCm instalado com sucesso!"
 
setup: install install-amd-torch
	@echo ""
	@echo ">>> Setup completo! Rode 'make test-gpu' para verificar a GPU."
 
test-gpu:
	@echo ">>> Testando reconhecimento da GPU AMD..."
	@if [ -z "$(VENV_PATH)" ]; then \
		echo "ERRO: Nenhum ambiente virtual Poetry encontrado."; \
		exit 1; \
	fi
	$(VENV_PATH)/bin/python -c "\
import torch; \
print('ROCm disponivel:', torch.cuda.is_available()); \
print('Num GPUs:', torch.cuda.device_count()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Nenhuma'); \
print('Versao HIP:', torch.version.hip); \
"
 