from ultralytics import YOLO
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def ecg_to_csv():

    model = YOLO("yolov8n.pt")

    image_path = '../teste.png'

    model.train(data="dataset.yaml", epochs=50, imgsz=480, device="cpu", workers=8)
    results = model.predict(image_path)

    # Filtrar bounding boxes com confiança >= 0.5
    filtered_boxes = []
    filtered_scores = []

    for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        if conf >= 0.5:
            filtered_boxes.append(box.numpy())  # converte tensor → numpy
            filtered_scores.append(float(conf))

    filtered_boxes = np.array(filtered_boxes)

    # Carregar imagem original
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Exibir cada lead recortada
    plt.figure(figsize=(15, 6))
    for i, box in enumerate(filtered_boxes):
        x1, y1, x2, y2 = map(int, box)
        lead_crop = img[y1:y2, x1:x2]

        plt.subplot(1, len(filtered_boxes), i+1)
        plt.imshow(lead_crop)
        plt.axis('off')
        plt.title(f'Lead {i+1}')

    plt.tight_layout()
    plt.show()

ecg_to_csv()