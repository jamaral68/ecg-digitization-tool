# Ecg-digitization-tool

## Summary
This project is a tool designed to extract individual ECG signals from scanned paper ECG records, so that the data can be recovered digitally and be used for further research or diagnoses.

## Installation

1. **Clone the repository**:

```bash

git clone -b matheus https://github.com/jamaral68/ecg-digitization-tool.git
cd ecg-digitization-tool

```

2. **Install the dependencies**:

```bash

pip install -r requirements.txt

```

3. **Run the main script**:

```bash

python main.py 

```

## Configuration Variables

| Variable | Description |
|----------|-------------|
| `image` | Path to the ECG image |
| `csv_name` | Output CSV file name |
| `pulse_per_sec` | Pulse width in seconds |
| `pulse_per_mv` | Pulse height in millivolts |
| `sample_frequency` | Sampling frequency (Hz) |
| `num_sampling_points` | Number of sampling points per lead |
| `hpulse` | Computed pulse height in pixels (internal use) |
| `wpulse` | Computed pulse width in pixels (internal use) |