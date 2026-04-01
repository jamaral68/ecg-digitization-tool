import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import segment_to_df, draw_overlay

def ecg_to_csv(setup, model: YOLO, save_overlay=True):
    results = model(setup.image)
    results[0].save()
    result = results[0]

    img = cv.imread(setup.image)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    pulse_boxes = [box for box in result.boxes if model.names[int(box.cls[0])].lower()=='pulse']
    if len(pulse_boxes) > 0:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].xyxy[0].tolist())
        pulse_height_px = y2 - y1
        pulse_per_mv = pulse_height_px / 1.0
        print(f"Pulse detected.")
    else:
        pulse_per_mv = 10.0
        print("Pulse not detected.")

    line_list = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]
        if lead_name.lower() == 'pulse':
            continue

        crop = img_gray[y1:y2, max(0,x1+10):min(img_gray.shape[1], x2-10)]
        height, width = crop.shape
        if width<2 or height<2:
            continue

        yseg = np.argmin(crop, axis=0)
        line_list.append({
            'wpulse': width,
            'hpulse': height,
            'curves':[{
                'xseg': np.arange(width),
                'yseg': yseg,
                'wseg': width,
                'name': lead_name
            }]
        })

    df = segment_to_df(line_list, setup.pulse_per_sec, pulse_per_mv, setup.num_sampling_points)

    # Save overlay
    if save_overlay:
        overlay_img = draw_overlay(setup.image, result, model)
        overlay_path = setup.csv_name.replace(".csv","_overlay.png")
        cv.imwrite(overlay_path, overlay_img)

    return df