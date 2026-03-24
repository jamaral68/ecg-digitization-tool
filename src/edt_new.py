import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

def ecg_to_csv(config):

    # Carregar imagem
    image = cv.imread(config)
    if image is None:
        print("Erro ao carregar imagem.")
        return

    print(f"INFO: Image Shape {image.shape}")

    # Converter BGR → RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Mostrar imagem original
    plt.imshow(image_rgb)
    plt.title("Imagem original")
    plt.axis("off")
    plt.show()

    # Carregar modelo (evite treinar toda vez!)
    model = YOLO("yolov8n.pt")

    model.train(data="dataset.yaml", epochs=200, imgsz=360, device="cpu", workers=8)

    # Predição
    results = model.predict(config)

    # Salvar imagem com bounding boxes
    results[0].save()

    # Extrair dados das boxes
    boxes_data = results[0].boxes

    if boxes_data is None:
        print("Nenhuma detecção encontrada.")
        return

    conf = boxes_data.conf.cpu().numpy()
    boxes = boxes_data.xyxy.cpu().numpy()

    # Filtrar por confiança mínima
    filtered = [(b, c) for b, c in zip(boxes, conf) if c > 0.5]

    if len(filtered) == 0:
        print("Nenhuma box com confiança suficiente.")
        return

    boxes = [b for b, _ in filtered]

    # Ordenar boxes (linha → coluna)
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    print(f"Total de segmentos detectados: {len(boxes)}")

    # Criar grid 4x3
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))

    axes = axes.flatten()

    for i in range(len(axes)):
        axes[i].axis("off")

    # Plotar segmentos
    for i, box in enumerate(boxes):
        if i >= 12:
            break

        x1, y1, x2, y2 = map(int, box)

        # Garantir que não sai da imagem
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_rgb.shape[1], x2)
        y2 = min(image_rgb.shape[0], y2)

        crop = image_rgb[y1:y2, x1:x2]

        axes[i].imshow(crop)
        axes[i].set_title(f"Lead {i+1}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# Caminho da imagem
config = '../teste.png'

ecg_to_csv(config)