import argparse
import torch
import cv2 as cv
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from edt import ecg_to_csv
from edt_utils import get_model, plot_ecg
from setup import Setup


def preprocess_image(image_path, device):
    img = cv.imread(image_path)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")

    img_tensor = F.to_tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB)).to(device)
    return img, img_tensor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract ECG signals from an image.")
    parser.add_argument("images", type=str, nargs='+')
    args = parser.parse_args()

    image = args.images[0]
    csv_name = image.replace(".png", ".csv")

    # =========================
    # Parâmetros ECG
    # =========================
    pulse_width_mm = 500
    mmpsec = 25
    pulse_per_sec = pulse_width_mm / mmpsec

    sample_frequency = 500
    time_lead = 2.5
    num_sampling_points = int(time_lead * sample_frequency)

    layout = (3, 4)

    lead_order = [
        'I', 'aVR', 'V1', 'V4',
        'II', 'aVL', 'V2', 'V5',
        'III', 'aVF', 'V3', 'V6',
    ]

    # =========================
    # Device
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # Modelos
    # =========================

    # Modelo de leads (14 classes)
    model_leads = get_model(num_classes=14)
    model_leads.load_state_dict(torch.load("CNN-leads.pth", map_location=device))
    model_leads.to(device)
    model_leads.eval()

    # Modelo de labels (2 classes = background + label)
    model_label = get_model(num_classes=2)
    model_label.load_state_dict(torch.load("CNN-labels.pth", map_location=device))
    model_label.to(device)
    model_label.eval()

    # =========================
    # Setup
    # =========================
    setup = Setup(
        image=image,
        csv_name=csv_name,
        pulse_per_sec=pulse_per_sec,
        sample_frequency=sample_frequency,
        num_sampling_points=num_sampling_points,
    )

    # =========================
    # Processamento ECG
    # =========================
    df = ecg_to_csv(
        setup,
        model_leads,
        device,
        label_model=model_label,
        save_overlay=True
    )

    # =========================
    # Plot
    # =========================
    fig = plot_ecg(
        df,
        lead_order,
        title=f"ECG - {image}",
        n_rows=layout[0],
        n_columns=layout[1],
        fs=setup.sample_frequency,
    )

    plt.show()

    # =========================
    # Save CSV
    # =========================
    df.to_csv(setup.csv_name, index=False)
    print(f"INFO: Saved {csv_name}")