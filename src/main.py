import argparse
import torch
from edt import ecg_to_csv
from edt_utils import get_model
from setup import Setup

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract ECG signals from an image.")
    parser.add_argument("images", type=str, nargs='+')
    args = parser.parse_args()

    image = args.images[0]
    csv_name = args.images[0]
    pulse_width_mm      = 500
    mmpsec              = 25
    pulse_per_sec       = pulse_width_mm / mmpsec
    sample_frequency    = 500
    time_lead           = 2.5
    num_sampling_points = int(time_lead * sample_frequency)
    layout              = (3, 4)

    lead_order = [
        'I',   'aVR', 'V1', 'V4',
        'II',  'aVL', 'V2', 'V5',
        'III', 'aVF', 'V3', 'V6',
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=14)
    model.load_state_dict(torch.load("faster_rcnn_ecg.pth", map_location=device))
    model.to(device)
    model.eval()
    
    setup = Setup(
            image=image,
            csv_name=csv_name,
            pulse_per_sec=pulse_per_sec,
            sample_frequency=sample_frequency,
            num_sampling_points=num_sampling_points,
        )

    df = ecg_to_csv(setup, model, device, label_model=None, save_overlay=True)