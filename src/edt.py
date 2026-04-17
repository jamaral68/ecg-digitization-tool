import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F
from edt_utils import (
    predict_and_draw,
    remove_labels_inpaint,
    segment_to_df
)

# Mapeamento manual de classes do modelo Faster R-CNN
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


def ecg_to_csv(setup, model, device, label_model=None, save_overlay=True):

    img = cv.imread(setup.image)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {setup.image}")

    # Mostrar imagem original
    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Imagem original")
    plt.axis("off")
    plt.show()

    # Visualização inicial (debug)
    img_out = predict_and_draw(model, img, device, threshold=0.5)

    plt.figure()
    plt.imshow(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    plt.title("Detecção ECG")
    plt.axis("off")
    plt.show()

    # Inferência Faster R-CNN
    model.eval()
    img_tensor = F.to_tensor(cv.cvtColor(img, cv.COLOR_BGR2RGB)).to(device)

    with torch.no_grad():
        result = model([img_tensor])[0]

    # YOLO (labels opcionais)
    label_boxes = []
    if label_model is not None:
        label_results = label_model(setup.image)
        for box in label_results[0].boxes:
            lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
            label_boxes.append((lx1, ly1, lx2, ly2))

    # Calibração (pulso)
    pulse_boxes = [
        box
        for box, label, score in zip(result['boxes'], result['labels'], result['scores'])
        if score >= 0.5 and CLASS_NAMES.get(int(label), '').lower() == 'pulse'
    ]

    if pulse_boxes:
        x1, y1, x2, y2 = map(int, pulse_boxes[0].tolist())
        pulse_per_mv = (y2 - y1) / 1.0
    else:
        pulse_per_mv = 10.0

    print(f"[INFO] pulse_per_mv = {pulse_per_mv}")

    # grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Extração das curvas
    line_list = []

    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):

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

        height, width = crop.shape

        if width < 2 or height < 2:
            continue

        # linha do ECG (mínimo por coluna)
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

    print(f"[INFO] Leads detectados: {[l['curves'][0]['name'] for l in line_list]}")

    # =========================
    # OVERLAY (Faster R-CNN)
    # =========================
    if save_overlay:
        overlay_img = img.copy()

        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            if score < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.tolist())
            name = CLASS_NAMES.get(int(label), str(label))

            # caixa verde (leads)
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

        # caixas vermelhas (labels YOLO)
        for (lx1, ly1, lx2, ly2) in label_boxes:
            cv.rectangle(overlay_img, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

        overlay_path = setup.csv_name.replace(".csv", "_overlay.png")
        cv.imwrite(overlay_path, overlay_img)

        plt.figure()
        plt.imshow(cv.cvtColor(overlay_img, cv.COLOR_BGR2RGB))
        plt.title("Overlay ECG (Faster R-CNN + Labels)")
        plt.axis("off")
        plt.show()

    # =========================
    # DataFrame final
    # =========================
    df = segment_to_df(
        line_list,
        setup.pulse_per_sec,
        pulse_per_mv,
        setup.num_sampling_points
    )

    return df