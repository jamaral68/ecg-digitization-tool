import argparse
import torch
import cv2 as cv
from edt_utils import get_model, predict_and_draw

if __name__ == "__main__":

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

    parser = argparse.ArgumentParser(description="Extract ECG signals from an image.")
    parser.add_argument("images", type=str, nargs='+')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=14)
    model.load_state_dict(torch.load("faster_rcnn_ecg.pth", map_location=device))
    model.to(device)
    model.eval()

    for img_path in args.images:
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        result = predict_and_draw(model, image, device)

        out_path = "result_" + img_path.split("/")[-1]
        cv.imwrite(out_path, cv.cvtColor(result, cv.COLOR_RGB2BGR))

        print(f"Saved: {out_path}")