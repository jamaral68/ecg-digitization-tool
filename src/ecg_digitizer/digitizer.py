import cv2 as cv
import numpy as np
from ultralytics import YOLO

from ecg_digitizer.config import DigitizerConfig
from ecg_digitizer.utils import draw_overlay, remove_labels_inpaint, segment_to_df


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
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    results = model(config.image)
    result = results[0]

    label_boxes = []
    if label_model is not None:
        label_results = label_model(config.image)
        for box in label_results[0].boxes:
            lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
            label_boxes.append((lx1, ly1, lx2, ly2))

    pulse_boxes = [
        box for box in result.boxes if model.names[int(box.cls[0])].lower() == "pulse"
    ]
    if pulse_boxes:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].xyxy[0].tolist())
        pulse_per_mv = (y2 - y1) / 1.0
    else:
        pulse_per_mv = 10.0

    line_list = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]

        if lead_name.lower() == "pulse":
            continue

        crop_x1 = max(0, x1 + 10)
        crop_x2 = min(img_gray.shape[1], x2 - 10)

        if label_boxes:
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
        line_list.append(
            {
                "wpulse": width,
                "hpulse": height,
                "curves": [
                    {
                        "xseg": np.arange(width),
                        "yseg": yseg,
                        "wseg": width,
                        "name": lead_name,
                    }
                ],
            }
        )

    if save_overlay:
        overlay_img = draw_overlay(config.image, result, model)
        for lx1, ly1, lx2, ly2 in label_boxes:
            cv.rectangle(overlay_img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        overlay_path = config.csv_name.replace(".csv", "_overlay.png")
        cv.imwrite(overlay_path, overlay_img)

    df = segment_to_df(
        line_list, config.pulse_per_sec, pulse_per_mv, config.num_sampling_points
    )
    return df
