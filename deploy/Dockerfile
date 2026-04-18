FROM public.ecr.aws/docker/library/python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-por \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install poetry==1.8.3

WORKDIR /opt/ml/code

COPY deploy/pyproject.sagemaker.toml pyproject.toml
COPY README.md ./
RUN poetry install --no-root --only main

RUN pip install jupyter jupyterlab ipykernel \
    && python -m ipykernel install --name ecg-digitization --display-name "Python (ECG)"

COPY src/ ./src/
COPY notebooks/ ./notebooks/

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
