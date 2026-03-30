"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import segment_to_df

def ecg_to_csv(setup):
    model = YOLO("best.pt")
    results = model(setup.image)
    result = results[0]

    line_list = []
    img = cv.imread(setup.image)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]
        if lead_name == 'pulse':
            continue

        crop = img_gray[y1:y2, x1+10:x2-10]
        height, width = crop.shape

        yseg = np.array([height - np.argmin(crop[:, col]) for col in range(width)])
        baseline = np.min(yseg)

        line_list.append({
            'wpulse': width,
            'hpulse': height,
            'curves': [{
                'xseg': np.arange(width),
                'yseg': yseg,
                'wseg': width,
                'baseline': baseline,
                'name': lead_name
            }]
        })

    df = segment_to_df(line_list, setup.pulse_per_sec, setup.pulse_per_mv, setup.num_sampling_points)

    return df