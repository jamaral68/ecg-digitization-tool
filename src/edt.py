import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import segment_to_df, draw_overlay


def ecg_to_csv(setup, model: YOLO, save_overlay=True):
    """
    Extract ECG signals from an image using YOLO and return a DataFrame.
    Optionally saves an overlay image for debugging.
    """
    results = model(setup.image)
    results[0].save()
    result = results[0]

    img = cv.imread(setup.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {setup.image}")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    line_list = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]

        if lead_name.lower() == 'pulse':
            continue

        crop = img_gray[y1:y2, max(0, x1 + 10):min(img_gray.shape[1], x2 - 10)]
        height, width = crop.shape
        if width < 2 or height < 2:
            continue

        yseg = np.array([np.argmin(crop[:, col]) for col in range(width)])

        line_list.append({
            'wpulse': width,
            'hpulse': height,
            'curves': [{
                'xseg': np.arange(width),
                'yseg': yseg,
                'wseg': width,
                'name': lead_name
            }]
        })

    df = segment_to_df(
        line_list,
        setup.pulse_per_sec,
        setup.pulse_per_mv,
        setup.num_sampling_points
    )

    # Save overlay
    if save_overlay:
        overlay_img = draw_overlay(setup.image, result, model)
        overlay_path = setup.csv_name.replace(".csv", "_overlay.png")
        cv.imwrite(overlay_path, overlay_img)

    return df