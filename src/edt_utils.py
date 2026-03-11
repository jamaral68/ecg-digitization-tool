import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage, interpolate

def py_blockproc(A, blockdims, func=0):
    vr, hr = A.shape[0] // blockdims[0], A.shape[1] // blockdims[1]
    B = np.zeros((vr,hr))

    verts = np.vsplit(A, vr)
    for i in range(len(verts)):
       for j, v in enumerate(np.hsplit(verts[i], hr)):
          B[i,j]=(np.std(A[
             i * blockdims[0] : (i + 1) * blockdims[0],
             j * blockdims[1] : (j + 1) * blockdims[1]
            ]))
    return B

def laplacian_filter(img, kSize=3, gSize=3, alpha=1.0):
    input_is_bgr = len(img.shape) == 3

    if input_is_bgr: 
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Gaussian blur / low-pass filter
    gauss = cv.GaussianBlur(gray_img, (gSize, gSize), 0.0)

    # Edge detection / high-pass filter
    lpl = cv.Laplacian(gauss, cv.CV_32F, ksize=kSize)
    if input_is_bgr: 
        lpl  = cv.cvtColor(lpl, cv.COLOR_GRAY2BGR)

    # Image sharpening
    filtered_img = img.astype("float32") - alpha * lpl
    
    return np.clip(filtered_img, 0.0, 255.0).astype("uint8")

def extract_image(cropped_img, kSize2d=3, kSize1d=3):
    """
    Based on https://github.com/alphanumericslab/ecg-image-kit.
    """

    # Convert BGR to grayscale
    if len(cropped_img == 3):
        gray_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = cropped_img

    # Image sharpening
    lpl_img = laplacian_filter(gray_img, 3, 7, 0.5)

    # 1D blur
    box1d_img = cv.filter2D(lpl_img, cv.CV_32F, np.ones((1, kSize1d), np.float32) / kSize1d)

    # 2D blur
    box2d_img = cv.filter2D(lpl_img, cv.CV_32F, np.ones((kSize2d, kSize2d), np.float32) / (kSize2d * kSize2d))

    # All left/right neighbors
    h0 = np.array([[1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0],
                   [1.0, 0.0, 1.0]], np.float32) / 7.0
    lr_neigh_img = cv.filter2D(lpl_img, cv.CV_32F, h0)

    # All combined neighbors    
    h1 = np.array([1.0, 1.0, 1.0], np.float32) / 3.0
    z1 = cv.filter2D(lpl_img, cv.CV_32F, h1)

    h2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], np.float32) / 3.0
    z2 = cv.filter2D(lpl_img, cv.CV_32F, h2)

    h3 = np.array([[0.0, 0.0, 1.0], 
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]], np.float32) / 3.0
    z3 = cv.filter2D(lpl_img, cv.CV_32F, h3)

    all_neigh_img = np.maximum(np.maximum(z1, z2), z3)

    output_img = np.median([lpl_img,
                            box1d_img,
                            box2d_img,
                            lr_neigh_img,
                            all_neigh_img
                            ], axis=0)
    
    # Normalize the output image to 0-255
    output_img = cv.normalize(output_img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    return output_img

def get_values_from_img(roi):
    '''
    Get the x and y coordinates of the image containing the signal
    INPUT:
        roi: binary image with signal in white
    OUTPUT:
        xs, ys: signal values
    '''
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    width, length = roi.shape[:2]
    xs, ys = [], []
    bool_roi = roi != 0
    old_dy_dx = 0.0
    for i, col in enumerate(bool_roi.T):
        if len(xs) != 0:
            label, num = ndimage.label(col, structure=np.ones((3,)))
            if num != 0:
                median_list = []
                dy_dx_list = []
                d2y_dx2_list = []
                for n in range(1, num + 1):
                    pixel_loc = width - np.where(label == n)[0]
                    median_pixel = find_nearest(pixel_loc, np.median(pixel_loc))
                    median_list.append(median_pixel)
                    dy_dx = (median_pixel - ys[-1]) #/ (i - xs[-1])
                    d2y_dx2 = (dy_dx - old_dy_dx) #/ (i - xs[-1])
                    dy_dx_list.append(dy_dx)
                    d2y_dx2_list.append(d2y_dx2)
                tmp = np.argmin(np.abs(d2y_dx2_list))
                old_dy_dx = dy_dx_list[tmp]
                xs.append(i)
                ys.append(median_list[tmp])
        else:
            pixel_loc = width - np.where(col)[0]
            if pixel_loc.size > 0:
                median_pixel = np.median(pixel_loc)
                xs.append(i)
                ys.append(median_pixel)
    return width, length, xs, ys

def measure_extract_pulse(x, y, verbose=0):
    min_pulse = np.min(y)
    max_pulse = np.max(y)

    height = np.max(max_pulse-min_pulse)
    threshold = height / 2
    index = np.where((y - min_pulse)>=threshold)[0]
    width = x[index[-1]] - x[index[0]]
    if verbose > 0:
        print(f"pulse height: {height}")
        print(f"pulse width: {width} time units")
    return width, height

def detect_ref_pulse(roi, template,location='right', threshold=0.6, verbose=2):
    '''
    Detects a reference pulse using template matching
    '''
    if roi.shape[0] <= template.shape[0] or roi.shape[1] <= template.shape[1]:
        # template is bigger than ROI. Cannot perform matchTemplate
        empty_list=[]
        empty_array = np.array(empty_list)
        loc = (empty_array,empty_array )
    else:
        method = cv.TM_CCORR_NORMED
        res = cv.matchTemplate(roi,template,method) # try to find the pulse using template

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            x = top_left[1]
            y = top_left[0]
            similarity_value = min_val
            print("INFO: min similarity value is {} in x = {} and y = {}.".format(min_val,x,y))
        else:
            top_left = max_loc
            x = top_left[1]
            y = top_left[0]
            similarity_value = max_val
            print("INFO: max similarity value is {} in x = {} and y = {}.".format(max_val,x,y))

        # TODO: check if necessary
        x = top_left[1]
        y = top_left[0]

        template_width, template_height = template.shape
        if verbose > 1:
            plt.imshow(roi)
            rect = plt.Rectangle((y, x), template_height, template_width, color='red',
                    fc='none')
            plt.gca().add_patch(rect)
            plt.title('Grayscale image with bounding box around the pulse')
            plt.show()

        loc = np.where(res >= threshold)

    if len(loc[0])>0:
        detected = True # pulse detected

        ppts = np.array(list(map(list, zip(*loc[::-1])))) # obtain array from list of tuples

        ppts_max = ppts[:,0].max()
        ppts_min = ppts[:,0].min()
        ppts_median = np.median(ppts[:,0])

        extracted_pulse = roi[x:x+template_width, y:y+template_height]
        _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)

    else:
              # No pulse detected or ROI has no pulse
              detected = False
              wpulse = np.nan
              hpulse = np.nan

    return detected, location, similarity_value, x, y, wpulse, hpulse

def display_segments(name, item, axis='off'):
    plt.figure(figsize=(12, 9))
    plt.imshow(item, cmap="magma")
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)
    plt.show()

def process_line(line_number, labeled_line, offset, line_leads, config_dict, verbose):
    """
    Process a labeled ECG line, segmenting it into individual curves
    and calculating width, height, and baseline information for each segment.
    """

    # Dictionary to store line information
    line_dict = {}
    line_dict['wpulse'] = config_dict.wpulse
    line_dict['hpulse'] = config_dict.hpulse
    line_dict['curves'] = []
    line_dict['offset_line'] = offset

    display_title = "Labeled Line " + str(line_number)
    display_segments(display_title, labeled_line)

    u, c = np.unique(labeled_line, return_counts=True)
    segment_labels = np.argsort(-c[1:]) + 1  # sort labels by segment size (descending)
    segment_length = -np.sort(-c[1:])
    max_label = np.max(u)

    app_seg_size = labeled_line.shape[1] // config_dict.layout[1]

    if verbose > 1:
        print("INFO: unique labels: ", u)
        print("INFO: counts: ", c)
        print("INFO: segment labels: ", segment_labels)
        print("INFO: segment lengths: ", segment_length)

    temp = np.round(app_seg_size * 0.25, 0)

    # Iterate over all detected labels
    for label in segment_labels:
        roi = (labeled_line == label)
        sl = ndimage.find_objects(roi)

        if len(sl) == 0:
            continue

        roi = roi[sl[0][0], sl[0][1]]
        roi_length = roi.shape[1]

        if roi_length < temp:
            continue

        roi_copy = roi.astype(np.uint8) * 255

        # Calculate the ratio between segment length and approximate segment size
        ratio = round(roi_length / app_seg_size, 0)
        if verbose > 0:
            print(f"INFO: label={label}, length={roi_copy.shape[1]}, ratio={ratio}")

        # Ignore the rhythm line
        if line_number + 1 == config_dict.rhythm:
            continue

        # Internal function to create a segment dictionary
        def create_segment(start_x, stop_x, start_y, stop_y, seg_label):
            segment_dict = {
                'line': line_number,
                'label': seg_label,
                'start_x': start_x,
                'stop_x': stop_x,
                'start_y': start_y,
                'stop_y': stop_y
            }
            seg = labeled_line[start_x:stop_x, start_y:stop_y]
            seg = np.where(seg == label, 255, 0)

            ws, ls, xs, ys = get_values_from_img(seg)
            segment_dict['wseg'] = ws
            segment_dict['lseg'] = ls
            segment_dict['xseg'] = xs
            segment_dict['yseg'] = ys
            segment_dict['baseline'] = np.argmax(np.std(seg, axis=1))

            return segment_dict

        # Split segments according to the ratio
        if ratio >= 4:
            if verbose > 0:
                print("INFO: Four or more segments detected")
        elif ratio == 3:
            # Split into 3 segments
            x_start, x_stop = sl[0][0].start, sl[0][0].stop
            y_start, y_stop = sl[0][1].start, sl[0][1].stop
            step = (y_stop - y_start) // 3
            for i in range(3):
                seg = create_segment(x_start, x_stop, y_start + i*step, y_start + (i+1)*step if i<2 else y_stop, max_label)
                line_dict['curves'].append(seg)
                max_label += 1
        elif ratio == 2:
            # Split into 2 segments
            x_start, x_stop = sl[0][0].start, sl[0][0].stop
            y_start, y_stop = sl[0][1].start, sl[0][1].stop
            step = (y_stop - y_start) // 2
            for i in range(2):
                seg = create_segment(x_start, x_stop, y_start + i*step, y_start + (i+1)*step if i<1 else y_stop, max_label)
                line_dict['curves'].append(seg)
                max_label += 1
        elif ratio == 1:
            # Only one segment
            seg = create_segment(sl[0][0].start, sl[0][0].stop, sl[0][1].start, sl[0][1].stop, label)
            line_dict['curves'].append(seg)

    # Sort segments by y-position and add lead names
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])
    for i, d in enumerate(line_dict['curves']):
        if i < len(line_leads):
            d['name'] = line_leads[i]

    return line_dict

def is_nan(value):
    try:
        return math.isnan(float(value))
    except ValueError:
        return False

def print_segment_list(segment_list):
     for seg in segment_list:
          print("line number: {} - name: {} - segment length: {}".format (seg['line'],seg['name'] ,seg['lseg']))
          fig = plt.figure()
          plt.title(seg['name'])
          plt.plot(seg['xseg'], seg['yseg'])
          plt.grid()
          plt.show()

def print_line_dict(line):
     for key, value in line.items():
      if key =='curves':
           print_segment_list(value)
      else:
          print(f"{key}: {value}")

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