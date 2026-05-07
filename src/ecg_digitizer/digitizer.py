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
    extract_yseg_clean,
)
from ecg_scanner.scanner import ECGScanner

LEAD_ORDER = [
    "I",
    "aVR",
    "V1",
    "V4",
    "II",
    "aVL",
    "V2",
    "V5",
    "III",
    "aVF",
    "V3",
    "V6",
]

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

def crop_image_boxes(
    img,
    boxes,
    label_boxes,
    binarize_method="adaptive",
    block_size=50,
    block_threshold=25,
    pre_median_k=3,
    pre_gaussian_k=3,
    post_median_k=5,
):
    scanner = ECGScanner(
        v_margin=90, s_margin=60, fill_value=255, dark_percentile=10, s_quantile_offset=0.4
    )
    return scanner.scan_yolo(
        image=img,
        lead_boxes=boxes,
        label_boxes=label_boxes,
        binarize_method=binarize_method,
        block_size=block_size,
        block_threshold=block_threshold,
        pre_median_k=pre_median_k,
        pre_gaussian_k=pre_gaussian_k,
        post_median_k=post_median_k,
    )


def ecg_to_csv(
    config: DigitizerConfig,
    model: YOLO,
    label_model: YOLO | None = None,
    save_overlay: bool = True,
    binarize_method: str = "adaptive",
    block_size: int = 50,
    block_threshold: int = 25,
    pre_median_k: int = 3,
    pre_gaussian_k: int = 3,
    post_median_k: int = 5,
):
    """
    Extract ECG signals from an image and return a DataFrame.
    """
    img = cv.imread(config.image)
    results = model(config.image)
    result = results[0]

    boxes, pulse_per_mv = get_image_boxes(result=result, yolo_model=model)

    label_boxes = get_label_boxes(label_model, config)

    boxes = crop_image_boxes(
        img.copy(),
        boxes=boxes,
        label_boxes=label_boxes,
        binarize_method=binarize_method,
        block_size=block_size,
        block_threshold=block_threshold,
        pre_median_k=pre_median_k,
        pre_gaussian_k=pre_gaussian_k,
        post_median_k=post_median_k,
    )
    _ = boxes.pop("pulse")
    ecg_curves = dict()
    for lead_name, img_lead in boxes.items():
        height, width = img_lead.shape
        yseg = extract_yseg_clean(img_lead)
        ecg_curves[lead_name] = {
                "wpulse": width,
                "hpulse": height,
                "xseg": np.arange(width),
                "yseg": yseg,#np.argmin(img_lead,axis=0),
                "yseg_original":np.argmin(img_lead,axis=0),
                "wseg": width,
                "rec":boxes[lead_name]
            }
    lead_order_lower = [l.lower() for l in LEAD_ORDER]
    ordered_curves = {
        lead_lower: ecg_curves[lead_lower]
        for lead_lower in lead_order_lower
        if lead_lower in ecg_curves
    }
    ordered_curves.update({k: v for k, v in ecg_curves.items() if k not in lead_order_lower})
    
    df = segment_to_df(
        ordered_curves, config.pulse_per_sec, pulse_per_mv, config.num_sampling_points
    )
    return df
