import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from strategy import *
from edt_utils import *
from scipy.signal import find_peaks

def ecg_to_csv(config):

    # Reading and pre-processing the image
    image = cv.imread(config.image)
    
    print("INFO: pulse on lines: {}.".format(config.pulse))
    print("INFO: rhythm on line : {}.".format(config.rhythm))  
    print("INFO: Image Shape {}.".format(image.shape))
    plt.imshow(image)
    plt.show()

    # Thresholding 
    if config.strategy == 'color':
        ret, th1, image_gray = color(image, config.lower, config.upper, config.thres_value)
    elif config.strategy == 'filter':
        ret, th1, image_gray = filter(image, config.kSize2d, config.kSize1d, config.thres_value)
    elif config.strategy == 'none':
        ret, th1, image_gray = none(image, config.thres_value)

    print("INFO: gray scale image Shape {}.".format(image_gray.shape))
    plt.imshow(image_gray, cmap='gray')
    plt.show()

    print("INFO: Binary image Shape {}.".format(th1.shape))
    plt.imshow(th1, cmap='gray')
    plt.show()

    # Morphological dilation of the foreground
    if config.dilation != 0:    
        foreground = cv.morphologyEx(255 - th1, cv.MORPH_DILATE, np.ones((3,3)), iterations=config.dilation)
    else:
        foreground = 255 - th1

    print("INFO: Foreground image Shape {}.".format(foreground.shape))
    plt.imshow(foreground, cmap='gray')
    plt.show()

    # Load template image in grayscale
    template = cv.imread(config.template, cv.IMREAD_GRAYSCALE)

    # CORRECTION: was checking 'image', should check 'template'
    if template is None:
        print('INFO: Cannot open the template: ' + template)
        new_template = None
    else:
        # Apply thresholding to create template for pulse detection
        _, new_template = cv.threshold(template, 127, 255, cv.THRESH_OTSU)
        new_template = (new_template != 255) * np.uint8(255)

    print("INFO: Template image shape {}.".format(new_template.shape))
    plt.imshow(new_template, cmap = 'gray')
    plt.show()

    # Extract intensity profile using block processing
    temp = py_blockproc(foreground, (1, foreground.shape[1]), func=0)
    median_temp = np.median(temp.flatten())

    # Detect peaks (pulses) in the profile
    peak_indices, peak_dict = find_peaks(temp.flatten(), height=median_temp, distance=round(temp.flatten().size * config.perc_space_leads, 0))
    peak_heights = peak_dict['peak_heights']

    # Select the highest peaks
    highest_peak_index = peak_indices[np.argsort(peak_heights)]

    print("INFO: Plotting signal with highlighted peaks and median line")
    print(f"INFO: Total number of data points: {len(temp.flatten())}")
    print(f"INFO: Median value: {median_temp}")
    print(f"INFO: Peaks highlighted at indices: {highest_peak_index[-(config.layout[0]+1):]}")
    plt.plot(temp.flatten())
    # Highlight the peaks and median line
    plt.plot(highest_peak_index[-(config.layout[0]+1):], temp[highest_peak_index[-(config.layout[0]+1):]], "x")
    plt.plot(median_temp*np.ones_like(temp), "--", color="gray")
    plt.show()

    ordered_hp_index = sorted(highest_peak_index[-(config.layout[0] + 1):])

    # Calculate maximum distance to slice lines
    peak_dist = [np.abs(t - s) for s, t in zip(ordered_hp_index, ordered_hp_index[1:])]
    max_dist = int(np.round(max(peak_dist) * config.perc_max_dist, 0))

    # Create slices for processing each line
    slices_x = [(max(0, s - max_dist), min(foreground.shape[0], s + max_dist), None) for s in ordered_hp_index]
    slices_y = [(0, foreground.shape[1], None) for _ in ordered_hp_index]
    
    print("INFO: slices: {}". format(slices_x))

    # Create a list to store the processed lines
    proc_line_list = []

    # Extract and process the leads
    for i, slx in enumerate(slices_x): 

        line = foreground[slice(*slx),slice(*(0, foreground.shape[1], None))]
        offset = slx # reference to locate the segment in the image
        plt.imshow(line, cmap="gray")
        plt.show()
        structure = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)

        # Label connected segments in the line
        labeled_line, nb = ndimage.label(line, structure=structure)

        print("INFO: Number of segments {} on line {}.".format(nb, i))
        display_segments('Labeled line' + str(i), labeled_line)

        # Initialize default pulse width and height
        wt = 38  # default value
        ht = 75  # default value

        if (config.pulse == -1) or (i in config.pulse) :   # Check if the pulse is present
            line_signal = (labeled_line != 0) * np.uint8(255)

            # Try to detect the pulse
            line_copy = line_signal.copy()
            template_width, template_height = template.shape
            _, _, xt, yt = get_values_from_img(new_template)
            wt, ht = measure_extract_pulse(xt, yt, verbose=0)
            config.hpulse = ht # default values
            config.wpulse = wt

            # Pulse detection by template matching
            detected, location, similarity_value, x,y, wpulse, hpulse= detect_ref_pulse(line_copy, new_template)
            print("INFO: line {}: best similarity value = {} in {}". format(i,similarity_value,y))

            if detected :
                if  location =='right':
                    sliced_labeled_line = labeled_line[:,0:y].copy()
                elif location == 'left':
                    sliced_labeled_line = labeled_line[:,y+int(wpulse):].copy()
                else:
                    sliced_labeled_line = labeled_line.copy()     
            else:
                 if is_nan(wpulse):
                    wpulse = wt
                    hpulse = ht
                 sliced_labeled_line = labeled_line.copy() 

            if detected:
                print('INFO: pulse detected by template in line {} in {}'.format(i,y))
                plt.imshow(line_copy[x:x+int(template_width)+1,y:y+int(template_height)+1], cmap ="gray")
                plt.show()
            else:
                print('INFO: pulse NOT detected by template in line {}'.format(i))            

        else:
            print("INFO: line {} has no pulse to detect".format(i))
            wpulse = wt
            hpulse = ht
            sliced_labeled_line = labeled_line.copy()

        # Update configuration with detected pulse info
        config.wpulse = wpulse
        config.hpulse = hpulse

        # Process the line into segments
        line_dict = process_line(i, sliced_labeled_line, offset, config.lt_leads[i], config, 2)
        proc_line_list.append(line_dict)

    # Print to check if everything is correct
    for i, line in enumerate(proc_line_list): 
        print("INFO: processing line {}".format(i))
        print_line_dict(line)

    # Remove the rhythm line from the list of lines if present
    if config.rhythm != 0:
        proc_line_list.pop(config.rhythm-1)

    # Convert processed segments to a dataframe
    ecg_df = segment_to_df(proc_line_list, config.pulse_per_sec, config.pulse_per_mv, config.num_sampling_points)
    ecg_df.to_csv(config.csv_name)
    
    return ecg_df