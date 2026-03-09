import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import ndimage
from edt_utils import py_blockproc
from setup import setup_ecg
from strategy import color, filter, none
from edt_utils import detect_ref_pulse, get_values_from_img, measure_extract_pulse, is_nan, display_segments, process_line, print_line_dict, segment_to_df

def load_image_stage(data):

    config = data["config"]

    lt_leads, image = setup_ecg(
        config['layout'],
        config['pulse'],
        config['rhythm'],
        data["image_name"]
    )

    data["image"] = image
    data["lt_leads"] = lt_leads

    return data

def binarize_stage(data):

    image = data["image"]
    config = data["config"]

    if config["strategy"] == "color":
        _, th1, gray = color(image, config["lower"], config["upper"], config["thres_value"])

    elif config["strategy"] == "filter":
        _, th1, gray = filter(image, config["kSized2d"], config["kSized1d"], config["thres_value"])

    else:
        _, th1, gray = none(image, config["thres_value"])

    data["th1"] = th1
    data["image_gray"] = gray

    return data

def foreground_stage(data):

    th1 = data["th1"]
    config = data["config"]

    if config["dilation"] != 0:
        foreground = cv.morphologyEx(
            255 - th1,
            cv.MORPH_DILATE,
            np.ones((3,3)),
            iterations=config["dilation"]
        )
    else:
        foreground = 255 - th1

    data["foreground"] = foreground

    return data

def detect_leads_stage(data):

    foreground = data["foreground"]
    config = data["config"]
    layout = config["layout"]
    verbose = config["verbose"]
    perc_space_leads = config["perc_space_leads"]

    # Vertical image profile
    temp = py_blockproc(
        foreground,
        (1, foreground.shape[1]),
        func=0
    )

    temp_flat = temp.flatten()

    # Reference value
    median_temp = np.median(temp_flat)

    # Detects spikes (possible leads)
    peak_indices, peak_dict = find_peaks(
        temp_flat,
        height=median_temp,
        distance=round(temp_flat.size * perc_space_leads)
    )

    peak_heights = peak_dict["peak_heights"]

    # Sort peaks by height
    highest_peak_index = peak_indices[np.argsort(peak_heights)]

    # Select only the necessary ones
    selected_peaks = highest_peak_index[-(layout[0] + 1):]

    if verbose > 0:

        plt.plot(temp_flat)

        plt.plot(
            selected_peaks,
            temp_flat[selected_peaks],
            "x"
        )

        plt.plot(
            median_temp * np.ones_like(temp_flat),
            "--",
            color="gray"
        )

        plt.title("Lead Detection")
        plt.show()

    # saves to pipeline
    data["temp_profile"] = temp_flat
    data["peak_indices"] = peak_indices
    data["peak_heights"] = peak_heights
    data["selected_peaks"] = selected_peaks

    return data

def segment_lines_stage(data):

    foreground = data["foreground"]
    selected_peaks = data["selected_peaks"]
    config = data["config"]
    verbose = config["verbose"]

    perc_max_dist = config["perc_max_dist"]

    # Sort detected peaks to ensure correct vertical order of ECG leads
    ordered_peaks = sorted(selected_peaks)

    # Compute the vertical distance between consecutive peaks
    # This distance represents the approximate spacing between ECG leads
    peak_dist = [
        abs(t - s)
        for s, t in zip(ordered_peaks, ordered_peaks[1:])
    ]

    # Define the cropping window around each peak
    # The window size is based on the maximum distance between peaks
    max_dist = int(np.round(max(peak_dist) * perc_max_dist, 0))

    # Create vertical slices that will extract each ECG line
    slices_x = [
        (
            max(0, peak - max_dist),                 # upper boundary
            min(foreground.shape[0], peak + max_dist), # lower boundary
            None
        )
        for peak in ordered_peaks
    ]

    # Horizontal slices always span the full image width
    slices_y = [
        (0, foreground.shape[1], None)
        for _ in ordered_peaks
    ]

    if verbose > 0:
        print("INFO: slices_x:", slices_x)

    # Store results in the pipeline data dictionary
    data["ordered_peaks"] = ordered_peaks
    data["slices_x"] = slices_x
    data["slices_y"] = slices_y
    data["max_dist"] = max_dist

    return data

def pulse_detection_stage(data):

    foreground = data["foreground"]
    slices_x = data["slices_x"]
    slices_y = data["slices_y"]
    config = data["config"]

    template = data["template"]
    new_template = data["new_template"]

    pulse = config["pulse"]
    verbose = config["verbose"]

    processed_lines = []

    for i, slx in enumerate(slices_x):

        # Extract ECG line from the foreground image
        line = foreground[
            slice(*slx),
            slice(*slices_y[i])
        ]

        offset = slx

        if verbose > 1:
            plt.imshow(line, cmap="gray")
            plt.title(f"Line {i}")
            plt.show()

        # Label connected components
        structure = np.ones((3,3), dtype=np.uint8)

        labeled_line, nb = ndimage.label(line, structure=structure)

        if verbose > 0:
            print(f"INFO: Number of segments {nb} on line {i}")
            display_segments(f"Labeled line {i}", labeled_line)

        # Default pulse size
        wt = 38
        ht = 75

        # Check if this line contains the reference pulse
        if (pulse == -1) or (i in pulse):

            line_signal = (labeled_line != 0) * np.uint8(255)

            line_copy = line_signal.copy()

            # Estimate pulse size from template
            _, _, xt, yt = get_values_from_img(new_template)

            wt, ht = measure_extract_pulse(
                xt,
                yt,
                verbose=0
            )

            config["hpulse"] = ht
            config["wpulse"] = wt

            # Detect pulse using template matching
            detected, location, similarity_value, x, y, wpulse, hpulse = detect_ref_pulse(
                line_copy,
                new_template
            )

            print(f"INFO: line {i}: best similarity value = {similarity_value} in {y}")

            if detected:

                # Remove pulse region from the line
                if location == "right":

                    sliced_labeled_line = labeled_line[:, 0:y].copy()

                elif location == "left":

                    sliced_labeled_line = labeled_line[:, y + int(wpulse):].copy()

                else:

                    sliced_labeled_line = labeled_line.copy()

            else:

                # Use default pulse size if detection failed
                if is_nan(wpulse):

                    wpulse = wt
                    hpulse = ht

                sliced_labeled_line = labeled_line.copy()

            if verbose > 0:
                if detected:
                    print(f"INFO: pulse detected by template in line {i} at {y}")
                else:
                    print(f"INFO: pulse NOT detected by template in line {i}")

        else:

            print(f"INFO: line {i} has no pulse to detect")

            wpulse = wt
            hpulse = ht

            sliced_labeled_line = labeled_line.copy()

        # Store pulse size in config
        config["wpulse"] = wpulse
        config["hpulse"] = hpulse

        processed_lines.append({

            "line_index": i,
            "labeled_line": sliced_labeled_line,
            "offset": offset

        })

    # Store results in pipeline data
    data["processed_lines"] = processed_lines

    return data

def process_lines_stage(data):

    processed_lines = data["processed_lines"]
    config = data["config"]
    lt_leads = data["lt_leads"]

    verbose = config["verbose"]

    proc_line_list = []

    for line_data in processed_lines:

        i = line_data["line_index"]
        labeled_line = line_data["labeled_line"]
        offset = line_data["offset"]

        # Process the ECG line using the existing algorithm
        line_dict = process_line(
            i,
            labeled_line,
            offset,
            lt_leads[i],
            config,
            verbose
        )

        proc_line_list.append(line_dict)

    # Debug information to verify extracted signals
    if verbose > 0:

        for i, line in enumerate(proc_line_list):

            print(f"INFO: processing line {i}")
            print_line_dict(line)

    # Remove rhythm lead if required by configuration
    if config["rhythm"] != 0:

        rhythm_index = config["rhythm"] - 1

        if rhythm_index < len(proc_line_list):

            proc_line_list.pop(rhythm_index)

    # Store results in pipeline
    data["proc_line_list"] = proc_line_list

    return data

def dataframe_stage(data):

    config = data["config"]

    ecg_df = segment_to_df(
        data["proc_line_list"],
        config["pulse_per_sec"],
        config["pulse_per_mv"],
        config["num_sampling_points"]
    )

    data["ecg_df"] = ecg_df

    return data