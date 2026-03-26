from ultralytics import YOLO
import cv2 as cv

"""
def train_model(model, data, epochs, imgsz, device, workers):
    
    model = YOLO(model)
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, workers=workers)

    return model 

"""

def ecg_to_csv():
    model = YOLO("best.pt")
    results = model("../teste.png")

    img = cv.imread("../teste.png")

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        crop = img[y1:y2, x1:x2]

        

ecg_to_csv()