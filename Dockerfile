
# ROCm/PyTorch latest
ARG BASE_IMAGE=rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
    && rm -rf /var/lib/apt/lists/*
    
# Pacotes Python 
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      numpy pandas matplotlib scipy opencv-python pillow pytesseract boto3 scikit-image && \
    python -m pip install \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1
    
WORKDIR /workspace
