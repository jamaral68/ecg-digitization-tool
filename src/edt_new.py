import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import plot_ecg, segment_to_df

"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""

image = '../ecg100.png'
pulse_width_mm = 5
pulse_height_mm = 10
mmpsec = 25
mmpmv = 10
pulse_per_sec = pulse_width_mm / mmpsec
pulse_per_mv = pulse_height_mm / mmpmv
sample_frequency = 500
time_lead = 2.5
num_sampling_points = int(time_lead * sample_frequency)
layout = (3, 4)

def ecg_to_csv(img_path=image):
    model = YOLO("best.pt")
    results = model(img_path)
    result = results[0]
    results[0].save()

    line_list = []
    img = cv.imread(img_path)
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

    df = segment_to_df(line_list, pulse_per_sec, pulse_per_mv, num_sampling_points)

    lead_order = ['I', 'aVR', 'V1', 'V4',
                  'II', 'aVL', 'V2', 'V5',
                  'III', 'aVF', 'V3', 'V6']

    for lead in lead_order:
        if lead not in df.columns:
            df[lead] = np.zeros(num_sampling_points)

    return df

# Exemplo de uso
df = ecg_to_csv()
plot_ecg(df, df.columns, 'ECG', n_rows=layout[0], n_columns=layout[1], fs=500, figure_size=(20, 12))