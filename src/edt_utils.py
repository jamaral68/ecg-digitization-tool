import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

def convert_to_secmv(xs, ys, wp, hp, ws, baseline, pulse_per_sec, pulse_per_mv):
    '''
    INPUTS:
        xs: x-axis in points
        ys: y-axis in points
        wp: pulse width in points
        hp: pulse height in points
        baseline: segment baseline in points
        ws: segment width in points
    '''
    zero_line = ws - baseline
    ymv = (ys - zero_line) / (hp * pulse_per_mv)
    sec_per_pts = (pulse_per_sec / wp)
    xsec = sec_per_pts * np.asarray(xs)
    return xsec, ymv

def interpolate_segment(x, y, num):
     x_interp = np.linspace(0.0, 1.0, len(x))
     f = interpolate.CubicSpline(x_interp, y)
     x_new = np.linspace(0.0, 1.0, int(num))
     y_new = f(x_new)
     return x_new, y_new

def segment_to_df(line_list, pulse_per_sec, pulse_per_mv,num_pts):
    '''
    INPUT:
    line_list
    pulse_per_sec
    pulse_per_mv
    num_pts: number of points after interpolation
    '''
    df = pd.DataFrame()

    # Check if there is at least one line
    for line in line_list:
        for seg in (line['curves']):
            xsec, ymv= convert_to_secmv(seg['xseg'], seg['yseg'], line['wpulse'],
                                        line['hpulse'], seg['wseg'], seg['baseline'],
                                        pulse_per_sec, pulse_per_mv)
            x_new, y_new =  interpolate_segment(xsec, ymv, num_pts)
            df[seg['name']] = y_new
    return df

def plot_ecg(df,columns,title, n_rows = 4, n_columns = 4, fs = 500, figure_size = (20, 12)):
    if (n_rows * n_columns) < len(columns):
        raise Exception('Columns must be equal to or smaller than the number of rows and columns.')
    fig, axes = plt.subplots(n_rows, n_columns, figsize = figure_size)
    fig.suptitle(title,fontsize = 20)
    
    for index,col in enumerate(columns):
        
        if n_rows == 1 or n_columns == 1:
            current_ax = axes[index]
        else:
            row_index = int(index/n_columns)
            col_index = int(index - n_columns*row_index)
            ax = axes[row_index][col_index]
            
        signal = df[col]
        ts = np.arange(signal.size) / fs

        ax= plot_ecg_signal(ts, signal, ax)
        
    plt.subplots_adjust(top=0.92,hspace = 0.45,wspace = 0.5)    
    plt.show()
    return fig

def plot_ecg_signal(time, signal,ax):
    ax.plot(time, signal)
    # setup major and minor ticks
    min_t = int(np.min(time))
    max_t = round(np.max(time))
    major_ticks = np.arange(min_t, max_t+1)
    ax.set_xticks(major_ticks)
    # Turn on minor ticks
    ax.minorticks_on()
    # Major grid
    ax.grid(which='major', linestyle='-', color='red', linewidth='1.0')
    # Minor grid
    ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')
    return ax