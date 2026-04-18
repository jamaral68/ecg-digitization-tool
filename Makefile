# ============================================================================
# Variaveis
# ============================================================================

AWS_REGION ?= sa-east-1
REPO_NAME  ?= ecg-digitization-tool
TAG        ?= latest
IMAGE      ?= $(REPO_NAME):$(TAG)
PORT       ?= 8888

.PHONY: help build run shell push deploy register deploy-full clean

help:
	@echo "Targets disponiveis:"
	@echo ""
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


install-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
	export PATH="$$HOME/.local/bin:$PATH"
	poetry install
