import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from lead import Lead 


"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""

def ecg_to_csv():
    model = YOLO("best.pt")
    results = model("../teste.png")
    result = results[0]

    img = cv.imread("../teste.png")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    leads_list = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = result.names[cls_id]

        lead = Lead(name=lead_name, coords=[x1, y1, x2, y2])
        crop = img_gray[y1:y2, x1+10:x2-10]

        # Converter crop em traçado xseg, yseg
        # Aqui usamos uma abordagem simples: para cada coluna, pegamos o ponto mínimo (ou máximo) do traçado
        height, width = crop.shape
        xseg = np.arange(width)
        yseg = []

        for col in range(width):
            column_data = crop[:, col]
            # detecta o traçado: assume que a linha é mais escura que o fundo
            y = np.argmin(column_data)  # pega o pixel mais escuro
            yseg.append(height - y)     # inverter eixo y para matplotlib

        leads_list.append({'line': cls_id, 'name': lead_name, 'xseg': xseg, 'yseg': yseg, 'lseg': len(xseg)})

    # Plotar cada lead como traçado
    for seg in leads_list:
        print("line number: {} - name: {} - segment length: {}".format(seg['line'], seg['name'], seg['lseg']))
        fig = plt.figure()
        plt.title(seg['name'])
        plt.plot(seg['xseg'], seg['yseg'])
        plt.grid()
        plt.show()


ecg_to_csv()