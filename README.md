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
python main.py ../teste.png

```


## Validation 

1. **Run script**:

```bash

cd tst
python validate_faster_rcnn.py \
  --weights <path_to_model.pth> \
  --images-dir <path_to_validation_images_folder> \
  --labels-dir <path_to_validation_labels_folder> \
  --num-classes <number_of_classes_excluding_background> \
  --class-names <path_to_class_names_txt_file> \
  --batch-size <number_of_images_per_batch>

```