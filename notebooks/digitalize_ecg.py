from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from ecg_scanner.scanner import ECGDigitizer

def digitalize_ecg(image_path):
    digitizer = ECGDigitizer()
    digitizer.scan(image_path)

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent / "datasets" / "sample_images"
    digitalize_ecg(base_path / "1026034238-0009.png")
