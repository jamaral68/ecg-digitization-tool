"""
# config ideal 
# name_model - yolov8s.pt
# epochs - 200
# imgsz - 768
# workers - 4
def train_model(name_model, data, epochs, imgsz, batch, device, workers):
    
    model = YOLO(name_model)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers
    )


    return model 

"""
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from edt_utils import segment_to_df

def ecg_to_csv(setup, model: YOLO):
    """
    Extract ECG signals from an image using YOLO and return a DataFrame.
    """
    # Run YOLO model on the input ECG image
    results = model(setup.image)
    results[0].save()
    result = results[0]

    # Load the image using OpenCV
    img = cv.imread(setup.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {setup.image}")
    
    # Convert the image to grayscale for easier processing
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    line_list = []  # List to store each lead's waveform data

    # Iterate over detected bounding boxes from YOLO
    for box in result.boxes:
        # Get the coordinates of the bounding box (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # Get the class ID of the detected object
        cls_id = int(box.cls[0])
        # Map class ID to lead name
        lead_name = model.names[cls_id]
        # Skip the 'pulse' detection used for reference lines
        if lead_name.lower() == 'pulse':
            continue

        # Crop the grayscale image to the bounding box region, leaving margins
        crop = img_gray[y1:y2, max(0, x1+10):min(img_gray.shape[1], x2-10)]
        height, width = crop.shape

        # Extract the waveform by finding the minimum pixel (signal) in each column
        yseg = np.array([height - np.argmin(crop[:, col]) for col in range(width)])
        # Estimate baseline using the 10th percentile of the waveform
        baseline = np.percentile(yseg, 10)

        # Append the lead waveform info to the line list
        line_list.append({
            'wpulse': width,           # Width of the cropped segment
            'hpulse': height,          # Height of the cropped segment
            'curves': [{
                'xseg': np.arange(width),  # X-axis positions of the waveform
                'yseg': yseg,              # Extracted waveform values
                'wseg': width,             # Segment width
                'baseline': baseline,      # Estimated baseline
                'name': lead_name          # Lead name
            }]
        })

    # Convert all extracted segments into a DataFrame with proper scaling
    df = segment_to_df(line_list, setup.pulse_per_sec, setup.pulse_per_mv, setup.num_sampling_points)
    return df