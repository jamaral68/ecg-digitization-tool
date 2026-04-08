import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import segment_to_df, draw_overlay, remove_labels_inpaint
 
 
def ecg_to_csv(setup, model: YOLO, label_model: YOLO = None, save_overlay=True):
    """
    Extract ECG signals from an image and return a DataFrame.
 
    """
    results = model(setup.image)
    results[0].save()
    result = results[0]
 
    img = cv.imread(setup.image)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # ── Collect label bounding boxes (absolute coords) ──────────────────────
    label_boxes = []
    if label_model is not None:
        label_results = label_model(setup.image)
        for box in label_results[0].boxes:
            lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
            label_boxes.append((lx1, ly1, lx2, ly2))
        print(f"Labels detected: {len(label_boxes)}")
 
    # ── Calibration pulse ────────────────────────────────────────────────────
    pulse_boxes = [
        box for box in result.boxes
        if model.names[int(box.cls[0])].lower() == 'pulse'
    ]
    if pulse_boxes:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].xyxy[0].tolist())
        pulse_per_mv = (y2 - y1) / 1.0
        print("Pulse detected.")
    else:
        pulse_per_mv = 10.0
        print("Pulse not detected.")
 
    # ── Per-lead extraction ──────────────────────────────────────────────────
    line_list = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]
 
        if lead_name.lower() == 'pulse':
            continue
 
        # Narrow crop slightly to avoid border artefacts
        crop_x1 = max(0, x1 + 10)
        crop_x2 = min(img_gray.shape[1], x2 - 10)
 
        if label_boxes:
            # Build a full-height slice from the lead's top, then trim below
            full_crop = remove_labels_inpaint(
                img_gray, label_boxes, x1_lead=crop_x1, y1_lead=y1
            )
            crop = full_crop[: (y2 - y1), : (crop_x2 - crop_x1)]
        else:
            crop = img_gray[y1:y2, crop_x1:crop_x2]
 
        height, width = crop.shape
        if width < 2 or height < 2:
            continue
 
        yseg = np.argmin(crop, axis=0)
        line_list.append({
            'wpulse': width,
            'hpulse': height,
            'curves': [{
                'xseg': np.arange(width),
                'yseg': yseg,
                'wseg': width,
                'name': lead_name,
            }]
        })
 
    df = segment_to_df(
        line_list, setup.pulse_per_sec, pulse_per_mv, setup.num_sampling_points
    )
 
    # ── Overlay ──────────────────────────────────────────────────────────────
    if save_overlay:
        overlay_img = draw_overlay(setup.image, result, model)
 
        # Draw label boxes in red on the overlay so they are visible
        for (lx1, ly1, lx2, ly2) in label_boxes:
            cv.rectangle(overlay_img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)
 
        overlay_path = setup.csv_name.replace(".csv", "_overlay.png")
        cv.imwrite(overlay_path, overlay_img)
 
    return df