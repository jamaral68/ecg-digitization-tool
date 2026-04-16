import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import interpolate
from scipy.signal import medfilt
 
def remove_labels_inpaint(img_gray: np.ndarray, label_boxes: list, x1_lead: int, y1_lead: int) -> np.ndarray:
    """
    Remove text labels from a grayscale ECG crop using inpainting.
 
    For each label bounding box that overlaps the lead crop region, a mask is
    created and cv2.inpaint reconstructs the underlying signal pixels.
 
    """
    crop = img_gray[y1_lead:, x1_lead:].copy()  # will be sliced later by caller
    h, w = crop.shape
    mask = np.zeros((h, w), dtype=np.uint8)
 
    for (lx1, ly1, lx2, ly2) in label_boxes:
        # Convert absolute coords to crop-local coords
        cx1 = max(lx1 - x1_lead, 0)
        cy1 = max(ly1 - y1_lead, 0)
        cx2 = min(lx2 - x1_lead, w)
        cy2 = min(ly2 - y1_lead, h)
        if cx1 >= cx2 or cy1 >= cy2:
            continue
        mask[cy1:cy2, cx1:cx2] = 255
 
    if mask.any():
        crop = cv.inpaint(crop, mask, inpaintRadius=7, flags=cv.INPAINT_TELEA)
 
    return crop
 
 
def convert_to_secmv(xs, ys, wp, pulse_per_sec, pulse_per_mv):
    """
    Convert ECG waveform pixel coordinates to physical units: time (seconds) and amplitude (mV).
    """
    ys_smooth = medfilt(ys, kernel_size=min(len(ys) // 2 * 2 + 1, 101))
    baseline_px = np.percentile(ys_smooth, 50)
    ymv = (baseline_px - ys) / pulse_per_mv
    xsec = np.array(xs) * (pulse_per_sec / wp)
    return xsec, ymv
 
 
def interpolate_segment(x, y, num):
    """
    Interpolate an ECG segment to a fixed number of points using cubic spline interpolation.
    """
    x_interp = np.linspace(0, 1, len(x))
    f = interpolate.CubicSpline(x_interp, y)
    x_new = np.linspace(0, 1, num)
    y_new = f(x_new)
    return x_new, y_new
 
 
def segment_to_df(line_list, pulse_per_sec, pulse_per_mv, num_pts):
    """
    Convert a list of ECG waveform segments into a pandas DataFrame.
    Each column represents one lead's interpolated ECG signal.
    """
    df = pd.DataFrame()
    for i, line in enumerate(line_list):
        for seg in line['curves']:
            xsec, ymv = convert_to_secmv(
                seg['xseg'], seg['yseg'],
                line['wpulse'], pulse_per_sec, pulse_per_mv
            )
            _, y_new = interpolate_segment(xsec, ymv, num_pts)
            col_name = seg['name']
            if col_name in df.columns:
                col_name = f"{col_name}_{i}"
            df[col_name] = y_new
    return df
 
def draw_overlay(image_path, result, model):
    """
    Draw ECG waveforms on top of the original image for visual validation.
    """
    img = cv.imread(image_path)
    overlay = img.copy()
 
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]
 
        if lead_name.lower() == 'pulse':
            continue
 
        crop = cv.cvtColor(img[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
        yseg = np.argmin(crop, axis=0)
        xseg = np.arange(len(yseg))
 
        color_map = {"I": (255, 0, 0), "II": (0, 255, 0), "III": (0, 0, 255)}
        color = color_map.get(lead_name, (0, 0, 255))
 
        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for k in range(len(xseg) - 1):
            pt1 = (x1 + int(xseg[k]),     y1 + int(yseg[k]))
            pt2 = (x1 + int(xseg[k + 1]), y1 + int(yseg[k + 1]))
            cv.line(overlay, pt1, pt2, color, 2)
 
    alpha = 0.7
    final = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return final
 
def plot_ecg_signal(time, signal, ax):
    """
    Plot a single ECG signal on a given Matplotlib axis with grid and axis labels.
    """
    ax.plot(time, signal)
    ax.set_xticks(np.arange(int(time[0]), round(time[-1]) + 1))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', color='red', linewidth=1.0)
    ax.grid(which='minor', linestyle=':', color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    return ax
 
 
def plot_ecg(df, columns, title, n_rows=4, n_columns=4, fs=500, figure_size=(20, 12)):
    """
    Plot multiple ECG leads from a DataFrame in a grid layout.
    """
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    fig.suptitle(title, fontsize=20)
 
    for idx, col in enumerate(columns):
        row_idx = idx // n_columns
        col_idx = idx % n_columns
        ax = axes[row_idx][col_idx] if n_rows > 1 and n_columns > 1 else axes[idx]
        ts = np.arange(df[col].size) / fs
        plot_ecg_signal(ts, df[col], ax)
 
    plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.5)
    return fig

def train_model(model, dataloader, optimizer, device, epochs=10):
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

def predict_and_draw(model, image, device, threshold=0.5):
    model.eval()
    
    img_tensor = torch.from_numpy(image / 255.).permute(2, 0, 1).float().to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    
    img_out = image.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        
        x1, y1, x2, y2 = map(int, box)
        
        cv.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(img_out, f"Lead {label}: {score:.2f}",
                    (x1, y1 - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    
    return img_out

def collate_fn(batch):
    return tuple(zip(*batch))