# Ecg-digitization-tool

## Summary
This project is a tool designed to extract individual ECG signals from scanned paper ECG records, converting them into digital data for further analysis, research, or clinical use.

The application uses a Streamlit interface combined with YOLO-based models to detect and reconstruct ECG signals from images.

## Installation

1. **Clone the repository**:

```bash

git clone -b matheus https://github.com/jamaral68/ecg-digitization-tool.git
cd ecg-digitization-tool

```

2. **Create and Activate the Conda Environment**:

```bash

conda env create -f environment.yml
conda activate ecg_env

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