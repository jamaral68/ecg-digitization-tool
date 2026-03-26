import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from strategy import *
from edt_utils import *
from scipy.signal import find_peaks
from ultralytics import YOLO
from scipy import ndimage

def ecg_to_csv(config):

    model = YOLO("yolov8n.pt")

    # Train model
    model.train(data="dataset.yaml", epochs=1, imgsz=360, device="cpu", workers=8)

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

    if template is None:
        print('INFO: Cannot open the template')
        new_template = None
    else:
        _, new_template = cv.threshold(template, 127, 255, cv.THRESH_OTSU)
        new_template = (new_template != 255) * np.uint8(255)

    print("INFO: Template image shape {}.".format(new_template.shape))
    plt.imshow(new_template, cmap='gray')
    plt.show()

    # Extract intensity profile
    temp = py_blockproc(foreground, (1, foreground.shape[1]), func=0)
    median_temp = np.median(temp.flatten())

    peak_indices, peak_dict = find_peaks(
        temp.flatten(),
        height=median_temp,
        distance=round(temp.flatten().size * config.perc_space_leads, 0)
    )

    peak_heights = peak_dict['peak_heights']
    highest_peak_index = peak_indices[np.argsort(peak_heights)]

    plt.plot(temp.flatten())
    plt.plot(highest_peak_index[-(config.layout[0]+1):],
             temp[highest_peak_index[-(config.layout[0]+1):]], "x")
    plt.plot(median_temp*np.ones_like(temp), "--", color="gray")
    plt.show()

    ordered_hp_index = sorted(highest_peak_index[-(config.layout[0] + 1):])

    peak_dist = [np.abs(t - s) for s, t in zip(ordered_hp_index, ordered_hp_index[1:])]
    max_dist = int(np.round(max(peak_dist) * config.perc_max_dist, 0))

    slices_x = [(max(0, s - max_dist), min(foreground.shape[0], s + max_dist), None)
                for s in ordered_hp_index]

    proc_line_list = []

    for i, slx in enumerate(slices_x):

        line = foreground[slice(*slx), :]
        offset = slx

        plt.imshow(line, cmap="gray")
        plt.show()

        structure = np.ones((3,3), np.uint8)
        labeled_line, nb = ndimage.label(line, structure=structure)

        print("INFO: Number of segments {} on line {}.".format(nb, i))
        display_segments('Labeled line' + str(i), labeled_line)

        wt, ht = 38, 75

        if (config.pulse == -1) or (i in config.pulse):

            # Extract binary signal
            line_signal = (labeled_line != 0) * np.uint8(255)

            line_copy = line_signal.copy()

            # Estimate pulse size from template
            _, _, xt, yt = get_values_from_img(new_template)
            wt, ht = measure_extract_pulse(xt, yt, verbose=0)

            config.hpulse = ht
            config.wpulse = wt

            line_copy_rgb = cv.cvtColor(line_copy, cv.COLOR_GRAY2RGB)
            line_copy_inverted = 255 - line_copy_rgb

            # YOLO-based pulse detection
            detected, location, similarity_value, x, y, wpulse, hpulse, result = detect_ref_pulse_yolo(
                line_copy_inverted, model
            )

            result_plot = result.plot()

            print("INFO: line {}: similarity = {} at {}".format(i, similarity_value, y))

            # Correct processing: always use labeled_line for slicing
            if detected:
                if location == 'right':
                    sliced_labeled_line = labeled_line[:, 0:y].copy()
                elif location == 'left':
                    sliced_labeled_line = labeled_line[:, y + int(wpulse):].copy()
                else:
                    sliced_labeled_line = labeled_line.copy()
            else:
                if is_nan(wpulse):
                    wpulse = wt
                    hpulse = ht
                sliced_labeled_line = labeled_line.copy()

            # Debug visualization
            if detected:
                plt.imshow(result_plot)
                plt.title("pulse detected")
                plt.axis('off')
                plt.show()
            else:
                print('INFO: pulse NOT detected')

        else:
            print("INFO: line {} has no pulse".format(i))
            wpulse = wt
            hpulse = ht
            sliced_labeled_line = labeled_line.copy()

        config.wpulse = wpulse
        config.hpulse = hpulse

        line_dict = process_line(
            i,
            sliced_labeled_line,
            offset,
            config.lt_leads[i],
            config
        )

        proc_line_list.append(line_dict)

    for i, line in enumerate(proc_line_list):
        print("INFO: processing line {}".format(i))
        print_line_dict(line)

    ecg_df = segment_to_df(
        proc_line_list,
        config.pulse_per_sec,
        config.pulse_per_mv,
        config.num_sampling_points
    )
    print(ecg_df)
    ecg_df.to_csv(config.csv_name)

    return ecg_df