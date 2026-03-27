"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def ecg_to_csv(setup):
    # Carrega o modelo
    model = YOLO("best.pt")
    
    # Faz a detecção
    results = model(setup.image)
    
    # Carrega a imagem original com OpenCV
    img = cv2.imread(setup.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte BGR para RGB
    
    # Itera sobre as detecções
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Recorta a região da bounding box
            crop = img[y1:y2, x1:x2]
            
            # Exibe o recorte
            plt.figure()
            plt.imshow(crop)
            plt.title(f"Detecção {i+1}")
            plt.axis('off')
            plt.show()


# Exemplo de uso
ecg_to_csv()