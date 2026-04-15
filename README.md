# Ecg-digitization-tool

## Summary
This project is a tool designed to extract individual ECG signals from scanned paper ECG records, converting them into digital data for further analysis, research, or clinical use.

## Installation

1. **Clone the repository**:

```bash

git clone -b matheus https://github.com/jamaral68/ecg-digitization-tool.git
cd ecg-digitization-tool

```

2. **Create and Activate the Conda Environment**:

```bash

# CPU
conda env create -f env_cpu.yml
conda activate ecg_env_cpu

# GPU
conda env create -f env_gpu.yml
conda activate ecg_env_gpu

```

3. **Run the main script**:

```bash

cd src
streamlit run app.py

```

4. **Open your browser (usually automatically) at**:

```bash

http://localhost:8501

```