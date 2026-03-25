from ultralytics import YOLO

def ecg_to_csv():
    
    model = YOLO("yolov8n.pt")

    model.train(data="dataset.yaml", epochs=100, imgsz=1024, device="cpu", workers=8)

    # Predição
    results = model.predict('../teste.png')

    # Salvar imagem com bounding boxes
    results[0].save()

ecg_to_csv()