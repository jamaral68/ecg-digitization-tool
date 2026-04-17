import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from edt_utils import predict_and_draw, remove_labels_inpaint, segment_to_df

CLASS_NAMES = {
    0: 'background',
    1: 'pulse',
    2: 'I',
    3: 'II',
    4: 'III',
    5: 'aVR',
    6: 'aVL',
    7: 'aVF',
    8: 'V1',
    9: 'V2',
    10: 'V3',
    11: 'V4',
    12: 'V5',
    13: 'V6',
}


def ecg_to_csv(setup, model_leads, device, label_model=None, save_overlay=True):

    img = cv.imread(setup.image)

    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Image original")
    plt.axis("off")
    plt.show()


    img_out = predict_and_draw(model_leads, img, device, threshold=0.5)

    plt.figure()
    plt.imshow(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    plt.title("ECG Detection (Leads)")
    plt.axis("off")
    plt.show()

    # =========================
    # Faster R-CNN inference (LEADS)
    # =========================
    model_leads.eval()

    img_tensor = F.to_tensor(
        cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ).to(device)

    with torch.no_grad():
        result_leads = model_leads([img_tensor])[0]

    # =========================
    # Faster R-CNN inference (LABELS MODEL)
    # =========================
    label_boxes = []

    if label_model is not None:
        label_model.eval()

        with torch.no_grad():
            result_labels = label_model([img_tensor])[0]

        for box, score in zip(result_labels['boxes'], result_labels['scores']):
            if score < 0.5:
                continue

            lx1, ly1, lx2, ly2 = map(int, box.tolist())
            label_boxes.append((lx1, ly1, lx2, ly2))

    pulse_boxes = [
        box
        for box, label, score in zip(
            result_leads['boxes'],
            result_leads['labels'],
            result_leads['scores']
        )
        if score >= 0.5 and CLASS_NAMES.get(int(label), '').lower() == 'pulse'
    ]

    if pulse_boxes:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].tolist())
        pulse_per_mv = (y2 - y1) / 1.0
    else:
        pulse_per_mv = 10.0

    print(f"[INFO] pulse_per_mv = {pulse_per_mv}")

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    line_list = []

    for box, label, score in zip(
        result_leads['boxes'],
        result_leads['labels'],
        result_leads['scores']
    ):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        lead_name = CLASS_NAMES.get(int(label), str(label))

        if lead_name.lower() == 'pulse':
            continue

        crop_x1 = max(0, x1 + 10)
        crop_x2 = min(img_gray.shape[1], x2 - 10)

        if label_boxes:
            full_crop = remove_labels_inpaint(
                img_gray,
                label_boxes,
                x1_lead=crop_x1,
                y1_lead=y1
            )
            crop = full_crop[: (y2 - y1), : (crop_x2 - crop_x1)]
        else:
            crop = img_gray[y1:y2, crop_x1:crop_x2]

        if crop.size == 0:
            continue

        h, w = crop.shape

        if w < 2 or h < 2:
            continue

        yseg = np.argmin(crop, axis=0)

        line_list.append({
            'wpulse': w,
            'hpulse': h,
            'curves': [{
                'xseg': np.arange(w),
                'yseg': yseg,
                'wseg': w,
                'name': lead_name,
            }]
        })

    print(f"[INFO] Leads detectados: {[l['curves'][0]['name'] for l in line_list]}")

    if save_overlay:
        overlay_img = img.copy()

        # leads (verde)
        for box, label, score in zip(
            result_leads['boxes'],
            result_leads['labels'],
            result_leads['scores']
        ):
            if score < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            name = CLASS_NAMES.get(int(label), str(label))

            cv.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(
                overlay_img,
                name,
                (x1, y1 - 5),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        for (lx1, ly1, lx2, ly2) in label_boxes:
            cv.rectangle(overlay_img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        overlay_path = setup.csv_name.replace(".csv", "_overlay.png")
        cv.imwrite(overlay_path, overlay_img)

        plt.figure()
        plt.imshow(cv.cvtColor(overlay_img, cv.COLOR_BGR2RGB))
        plt.title("Overlay ECG (Leads + Labels - Faster R-CNN)")
        plt.axis("off")
        plt.show()


    df = segment_to_df(
        line_list,
        setup.pulse_per_sec,
        pulse_per_mv,
        setup.num_sampling_points
    )

    return df