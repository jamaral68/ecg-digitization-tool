import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.utils import (
    draw_overlay,
    segment_to_df,
    draw_overlay_from_curves,
    line_list_to_curves_df,
    extract_curve_robust,
)
from ecg_scanner.scanner import ECGScanner



def get_image_boxes(result, yolo_model):
    boxes = dict()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = yolo_model.names[cls_id]
        boxes[lead_name.lower()] = np.array(
            [
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2],  # bottom-left
            ]
        )
    pulse_boxes = [
            box for box in result.boxes if yolo_model.names[int(box.cls[0])].lower() == "pulse"
        ]
    pulse_per_mv = 10.0
    if pulse_boxes:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].xyxy[0].tolist())
        pulse_per_mv = (y2 - y1) / 1.0
    
    return boxes, pulse_per_mv

def get_label_boxes(label_model, config):
    label_boxes = []
    if label_model is None:
        return label_boxes
    label_results = label_model(config.image)
    for box in label_results[0].boxes:
        lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
        label_boxes.append((lx1, ly1, lx2, ly2))
    return label_boxes

def crop_image_boxes(img, boxes, label_boxes):
    scanner = ECGScanner(
        v_margin=90, s_margin=60, fill_value=255, dark_percentile=10, s_quantile_offset=0.4
    )
    return scanner.scan_yolo(image=img, lead_boxes=boxes, label_boxes=label_boxes)


def ecg_to_csv(
    config: DigitizerConfig,
    model: YOLO,
    label_model: YOLO | None = None,
    save_overlay: bool = True,
):
    """
    Extract ECG signals from an image and return a DataFrame.
    """
    img = cv.imread(config.image)
    results = model(config.image)
    result = results[0]

    boxes, pulse_per_mv = get_image_boxes(result=result, yolo_model=model)

    label_boxes = get_label_boxes(label_model, config)

    boxes = crop_image_boxes(img.copy(), boxes=boxes, label_boxes=label_boxes)
    _ = boxes.pop("pulse")
    ecg_curves = dict()
    for lead_name, img_lead in boxes.items():
        height, width = img_lead.shape
        #yseg = extract_curve_robust(img_lead)
        yseg = np.argmin(img_lead,axis=0)
        ecg_curves[lead_name] = {
                "wpulse": width,
                "hpulse": height,
                "xseg": np.arange(width),
                "yseg": yseg,
                "wseg": width,
                "rec":boxes[lead_name]
            }
    #if save_overlay:
    #    overlay_img = draw_overlay_from_curves(config.image, line_list_to_curves_df(line_list), )
    #    for lx1, ly1, lx2, ly2 in label_boxes:
    #        cv.rectangle(overlay_img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        #overlay_path = config.csv_name.replace(".csv", "_overlay.png")
        #cv.imwrite(overlay_path, overlay_img)

    df = segment_to_df(
        ecg_curves, config.pulse_per_sec, pulse_per_mv, config.num_sampling_points
    )
    return df
