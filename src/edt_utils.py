import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

def convert_to_secmv(xs, ys, wp, hp, baseline, pulse_per_sec, pulse_per_mv):
    """
    Convert pixel coordinates to seconds (x) and millivolts (y).
    Each vertical square = 1 mV
    Each horizontal square = pulse_per_sec (pixels / s)
    """
    if hp == 0 or wp == 0:
        raise ValueError("hpulse and wpulse must be > 0 for conversion")

    # Use median of the waveform as baseline in pixels
    baseline_px = np.median(ys)
    
    # Vertical scaling: pixels to mV
    ymv = (ys - baseline_px) / pulse_per_mv

    # Horizontal scaling: pixels to seconds
    sec_per_px = pulse_per_sec / wp
    xsec = sec_per_px * np.asarray(xs)

    return xsec, ymv

def interpolate_segment(x, y, num):
    """
    Interpolate the segment to a fixed number of points using cubic spline.
    """
    # Normalize x-axis for spline interpolation
    x_interp = np.linspace(0.0, 1.0, len(x))
    f = interpolate.CubicSpline(x_interp, y)
    # Generate new evenly spaced x-values
    x_new = np.linspace(0.0, 1.0, int(num))
    # Interpolate y-values
    y_new = f(x_new)
    return x_new, y_new

def segment_to_df(line_list, pulse_per_sec, pulse_per_mv, num_pts):
    """
    Convert a list of waveform segments into a pandas DataFrame.
    Each column corresponds to a lead's interpolated ECG signal.
    """
    df = pd.DataFrame()
    for i, line in enumerate(line_list):
        for seg in line['curves']:
            # Convert pixel values to seconds and mV
            xsec, ymv = convert_to_secmv(
                seg['xseg'], seg['yseg'], line['wpulse'], line['hpulse'],
                seg['baseline'], pulse_per_sec, pulse_per_mv
            )
            # Interpolate to fixed number of points
            _, y_new = interpolate_segment(xsec, ymv, num_pts)
            col_name = seg['name']
            # Avoid overwriting duplicate lead names
            if col_name in df.columns:
                col_name = f"{col_name}_{i}"
            df[col_name] = y_new
    return df

def plot_ecg_signal(time, signal, ax):
    """
    Plot a single ECG signal on a given axis with grid and labels.
    """
    ax.plot(time, signal)
    # Major and minor grid lines
    min_t, max_t = int(np.min(time)), round(np.max(time))
    ax.set_xticks(np.arange(min_t, max_t+1))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', color='red', linewidth=1.0)
    ax.grid(which='minor', linestyle=':', color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (mV)')
    return ax

def plot_ecg(df, columns, title, n_rows=4, n_columns=4, fs=500, figure_size=(20,12)):
    """
    Plot multiple ECG leads from a DataFrame in a grid layout.
    
    df: DataFrame containing ECG signals
    columns: list of column names to plot (lead order)
    title: figure title
    n_rows, n_columns: subplot grid size
    fs: sampling frequency (Hz)
    figure_size: matplotlib figure size
    """
    if n_rows * n_columns < len(columns):
        raise ValueError("Insufficient subplots for the number of columns provided")
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    fig.suptitle(title, fontsize=20)

    for idx, col in enumerate(columns):
        # Determine the appropriate axis
        if n_rows == 1 or n_columns == 1:
            ax = axes[idx]
        else:
            row_idx = idx // n_columns
            col_idx = idx % n_columns
            ax = axes[row_idx][col_idx]
        # Time vector based on sampling frequency
        ts = np.arange(df[col].size) / fs
        # Plot the signal
        plot_ecg_signal(ts, df[col], ax)
    plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.5)
    return fig