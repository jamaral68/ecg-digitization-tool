import cv2 as cv
import numpy as np
from ultralytics import YOLO
import pandas as pd
from edt_utils import segment_to_df, plot_ecg
from lead import Lead

"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""

def ecg_to_csv(img_path="../teste.png", pulse_per_sec=1000, pulse_per_mv=2, num_pts=500):
    model = YOLO("best.pt")
    results = model(img_path)
    result = results[0]

    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    line_list = []
    lead_names = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = result.names[cls_id]

        crop = img_gray[y1:y2, x1+10:x2-10]
        height, width = crop.shape
        xseg = np.arange(width)
        yseg = []

        for col in range(width):
            column_data = crop[:, col]
            y = np.argmin(column_data)
            yseg.append(height - y)

        if lead_name != 'pulse':
            line = {
                'wpulse': width,
                'hpulse': max(yseg),
                'curves': [
                    {
                        'xseg': xseg,
                        'yseg': yseg,
                        'wseg': width,
                        'baseline': min(yseg),
                        'name': lead_name
                    }
                ]
            }
            line_list.append(line)
            lead_names.append(lead_name)  

    df = segment_to_df(line_list, pulse_per_sec, pulse_per_mv, num_pts)

    df_leads = pd.DataFrame([lead_names], columns=df.columns)
    df_leads.index = ['lead_name']

    df = pd.concat([df_leads, df], ignore_index=False)
    return df

# Exemplo de uso
layout = (3, 4)
df = ecg_to_csv()
plot_ecg(df,df.columns,'FUNCIONOU', n_rows = layout[0], n_columns = layout[1], fs = 500, figure_size = (20, 12))