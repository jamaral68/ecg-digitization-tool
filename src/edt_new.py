import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO

def ecg_to_csv(config):

    #print("INFO: pulse on lines: {}.".format(config.pulse))
    #print("INFO: rhythm on line : {}.".format(config.rhythm))  
    #print("INFO: Image Shape {}.".format(image.shape))
    image = cv.imread(config)
    plt.imshow(image)
    plt.show()

    model = YOLO("yolov8n.pt")

    model.train(data="dataset.yaml", epochs=200, imgsz=360, device="cpu", workers=8)
    
    results = model.predict(config)    

    results[0].save()

config = '../teste.png'
ecg_to_csv(config)