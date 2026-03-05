import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_and_correct_rotation_robust(image_path, debug=False):
    # Read image
    image = cv2.imread(image_path)
    orig = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --- STEP 1: Detect the grid lines (red/orange filtering) ---
    # Red has two ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)


    # If grid is not red, try generic high-contrast edges
    if cv2.countNonZero(mask) < 100:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 5)

    # --- STEP 2: Morphological processing to enhance lines ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    # --- STEP 3: Detect lines with Hough ---
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
        
    #Draw the lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green color, thickness 2
    plt.imshow('Detected Lines', mask)
    plt.show()
    if lines is None:
        raise ValueError("No grid lines detected. Try adjusting preprocessing.")

    # --- STEP 4: Calculate angle ---
    angles = []
    for rho, theta in lines[:,0]:
        angle_deg = (theta * 180 / np.pi) - 90
        if -45 < angle_deg < 45:
            angles.append(angle_deg)

    if not angles:
        raise ValueError("No valid angles found.")

    median_angle = np.median(angles)

    if debug:
        print(f"Detected rotation angle: {median_angle:.2f} degrees")

    # --- STEP 5: Rotate image ---
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(orig, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # Debug plots
    if debug:
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1); plt.title("Original"); plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.subplot(1,3,2); plt.title("Mask"); plt.imshow(mask, cmap='gray'); plt.axis("off")
        plt.subplot(1,3,3); plt.title("Rotated"); plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)); plt.axis("off")
        plt.show()

    return rotated, median_angle

# Example usage
rotated_img, angle = detect_and_correct_rotation_robust("bucket/img20250226_10514068.png", debug=True)
cv2.imwrite("ecg_corrected.png", rotated_img)
