import argparse
import torch
import cv2 as cv
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from edt import ecg_to_csv
from edt_utils import get_model, plot_ecg
from setup import Setup

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract ECG signals from an image.")
    parser.add_argument("images", type=str, nargs='+')
    args = parser.parse_args()
    image = args.images[0] # Path to the input ECG image (first CLI argument)
    csv_name = image.replace(".png", ".csv") # Output CSV filename derived from input image name
    pulse_width_mm = 500 # Physical width of the ECG paper section in millimeters (used for calibration)
    mmpsec = 25 # Paper speed in mm per second 
    pulse_per_sec = pulse_width_mm / mmpsec # Conversion factor from pixels/mm to seconds (used for time scaling)
    sample_frequency = 500 # Desired sampling frequency for reconstructed ECG signal (Hz)
    time_lead = 2.5 # Time duration (in seconds) represented per lead segment
    num_sampling_points = int(time_lead * sample_frequency) # Total number of points used to resample each lead signal
    layout = (3, 4) # Grid layout of ECG leads (3 rows x 4 columns)
    lead_order = [
        'I', 'aVR', 'V1', 'V4',
        'II', 'aVL', 'V2', 'V5',
        'III', 'aVF', 'V3', 'V6',
    ] # Standard ECG lead arrangement used for visualization and ordering

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model leads 
    model_leads = get_model(num_classes=14)
    model_leads.load_state_dict(torch.load("CNN-leads.pth", map_location=device))
    model_leads.to(device)
    model_leads.eval()

    # Model labels 
    model_label = get_model(num_classes=2)
    model_label.load_state_dict(torch.load("CNN-labels.pth", map_location=device))
    model_label.to(device)
    model_label.eval()

    setup = Setup(
        image=image,
        csv_name=csv_name,
        pulse_per_sec=pulse_per_sec,
        sample_frequency=sample_frequency,
        num_sampling_points=num_sampling_points,
    )

    df = ecg_to_csv(
        setup,
        model_leads,
        device,
        label_model=model_label,
        save_overlay=True
    )

    fig = plot_ecg(
        df,
        lead_order,
        title=f"ECG - {image}",
        n_rows=layout[0],
        n_columns=layout[1],
        fs=setup.sample_frequency,
    )

    plt.show()

    df.to_csv(setup.csv_name, index=False)
    print(f"INFO: Saved {csv_name}")
    print("THE END")