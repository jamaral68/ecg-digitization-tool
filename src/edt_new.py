import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pandas as pd
from edt_utils import plot_ecg

"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""

def ecg_to_csv(img_path="../teste.png", pulse_per_sec=1000, pulse_per_mv=2, num_pts=1250):
    model = YOLO("best.pt")
    results = model(img_path)
    result = results[0]

    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    signals = {}

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])

        lead_name = model.names[cls_id]

        if lead_name == 'pulse':
            continue

        crop = img_gray[y1:y2, x1+10:x2-10]
        height, width = crop.shape

        yseg = []
        for col in range(width):
            column_data = crop[:, col]
            y = np.argmin(column_data)
            yseg.append(height - y)

        yseg = np.array(yseg)

        yseg = yseg - np.min(yseg)

        yseg = yseg / pulse_per_mv

        x_old = np.linspace(0, 1, len(yseg))
        x_new = np.linspace(0, 1, num_pts)
        y_resampled = np.interp(x_new, x_old, yseg)

        signals[lead_name] = y_resampled

    lead_order = ['I', 'aVR', 'V1', 'V4',
                  'II', 'aVL', 'V2', 'V5',
                  'III', 'aVF', 'V3', 'V6']

    df = pd.DataFrame({lead: signals.get(lead, np.zeros(num_pts)) for lead in lead_order})

    return df

# Exemplo de uso
layout = (3, 4)
df = ecg_to_csv()
plot_ecg(df,df.columns,'FUNCIONOU', n_rows = layout[0], n_columns = layout[1], fs = 500, figure_size = (20, 12))