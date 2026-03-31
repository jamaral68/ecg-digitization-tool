import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.signal import medfilt

def draw_overlay(image_path, result, model):
    """
    Draw extracted ECG signals on top of the original image for visual validation.
    """
    img = cv.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    overlay = img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]

        if lead_name.lower() == 'pulse':
            continue

        crop = cv.cvtColor(img[y1:y2, x1:x2], cv.COLOR_BGR2GRAY)
        height, width = crop.shape

        if width < 2 or height < 2:
            continue

        yseg = np.array([np.argmin(crop[:, col]) for col in range(width)])
        xseg = np.arange(width)

        color_map = {
            "I": (255, 0, 0),
            "II": (0, 255, 0),
            "III": (0, 0, 255),
        }
        color = color_map.get(lead_name, (0, 0, 255))

        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for k in range(len(xseg) - 1):
            pt1 = (x1 + int(xseg[k]), y1 + int(yseg[k]))
            pt2 = (x1 + int(xseg[k + 1]), y1 + int(yseg[k + 1]))
            cv.line(overlay, pt1, pt2, color, 2)

    alpha = 0.7
    final = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return final

def convert_to_secmv(xs, ys, wp, hp, pulse_per_sec, pulse_per_mv):
    """
    Convert pixel coordinates of an ECG segment to physical units: time (seconds) and amplitude (mV).
    """
    # Smooth the signal to reduce noise
    kernel_size = min(len(ys)//2*2+1, 101)  # ensure odd kernel size
    ys_smooth = medfilt(ys, kernel_size=kernel_size)

    # Estimate baseline as the median of the smoothed signal
    baseline_px = np.percentile(ys_smooth, 50)

    # Convert pixel values to mV, adjusting orientation
    ymv = (baseline_px - ys) / pulse_per_mv

    # Convert x-coordinates from pixels to seconds
    sec_per_px = pulse_per_sec / wp
    xsec = np.asarray(xs) * sec_per_px

    return xsec, ymv


def interpolate_segment(x, y, num):
    """
    Interpolate a waveform segment to a fixed number of points using cubic spline interpolation.
    """
    x_interp = np.linspace(0.0, 1.0, len(x))
    f = interpolate.CubicSpline(x_interp, y)
    x_new = np.linspace(0.0, 1.0, int(num))
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
                seg['xseg'], seg['yseg'], line['wpulse'], line['hpulse'], pulse_per_sec, pulse_per_mv
            )
            _, y_new = interpolate_segment(xsec, ymv, num_pts)
            col_name = seg['name']
            if col_name in df.columns:
                col_name = f"{col_name}_{i}"
            df[col_name] = y_new
    return df


def plot_ecg_signal(time, signal, ax):
    """
    Plot a single ECG signal on a given Matplotlib axis with grid and axis labels.
    """
    ax.plot(time, signal)
    min_t, max_t = int(np.min(time)), round(np.max(time))
    ax.set_xticks(np.arange(min_t, max_t + 1))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', color='red', linewidth=1.0)
    ax.grid(which='minor', linestyle=':', color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    return ax


def plot_ecg(df, columns, title, n_rows=4, n_columns=4, fs=500, figure_size=(20,12)):
    """
    Plot multiple ECG leads from a DataFrame in a grid layout.
    """
    if n_rows * n_columns < len(columns):
        raise ValueError("Insufficient subplots for the number of columns provided")

    fig, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    fig.suptitle(title, fontsize=20)

    for idx, col in enumerate(columns):
        if n_rows == 1 or n_columns == 1:
            ax = axes[idx]
        else:
            row_idx = idx // n_columns
            col_idx = idx % n_columns
            ax = axes[row_idx][col_idx]
        ts = np.arange(df[col].size) / fs
        plot_ecg_signal(ts, df[col], ax)

    plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.5)
    return fig