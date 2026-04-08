# Ecg-digitization-tool

## Summary
This project is a tool designed to extract individual ECG signals from scanned paper ECG records, so that the data can be recovered digitally and be used for further research or diagnoses.

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
python main.py 

```