import matplotlib.pyplot as plt
from edt import ecg_to_csv
from setup import Setup
from edt_utils import plot_ecg

if __name__ == "__main__":

    image = '../ecg4.jpg'         # Path to the ECG image
    template_name = '../pul4.png'       # Pulse template image
    csv_name = '../ecg_test2.csv'      # Output CSV filename
    strategy = 'none'                  # Preprocessing strategy (none/filter/color)
    thres_value = 127                  # Threshold value for binarization
    dilation = 10                      # Number of dilation iterations
    perc_space_leads = 0.2             # Percentage spacing between leads
    layout = (3, 4)                    # ECG layout: rows x columns
    perc_max_dist = 0.7                # Maximum distance percentage for line slicing
    rhythm = 4                         # Which line has the rhythm
    pulse = [0, 1, 2]                  # Lines that have pulses
    pulse_width_mm = 5                  # Pulse width in mm
    pulse_height_mm = 10                # Pulse height in mm
    mmpsec = 25                         # mm per second (time scaling)
    mmpmv = 10                          # mm per mV (voltage scaling)
    pulse_per_sec = pulse_width_mm / mmpsec
    pulse_per_mv = pulse_height_mm / mmpmv
    sample_frequency = 500              # Sampling frequency in Hz
    time_lead = 2.5                     # Duration of the segment in seconds
    num_sampling_points = time_lead / (1 / sample_frequency)
    location = 'right'                  # Location of the reference pulse
    lower = (0, 0, 0)                   # Lower color threshold (black)
    upper = (179, 255, 220)             # Upper color threshold (dark gray)
    kSize2d = 3                          # Kernel size for 2D filters
    kSize1d = 3                          # Kernel size for 1D filters

    # Create a setup object with the configuration, including template and CSV
    setup = Setup(
        image=image,
        template=template_name,
        csv_name=csv_name,
        strategy=strategy,
        thres_value=thres_value,
        dilation=dilation,
        perc_space_leads=perc_space_leads,
        layout=layout,
        perc_max_dist=perc_max_dist,
        pulse=pulse,
        rhythm=rhythm,
        mmpsec=mmpsec,
        mmpmv=mmpmv,
        pulse_width_mm=pulse_width_mm,
        pulse_height_mm=pulse_height_mm,
        pulse_per_sec=pulse_per_sec,
        pulse_per_mv=pulse_per_mv,
        sample_frequency=sample_frequency,
        time_lead=time_lead,
        location=location,
        num_sampling_points=num_sampling_points,
        lower=lower,
        upper=upper,
        kSize2d=kSize2d,
        kSize1d=kSize1d
    )

    # Call the main ECG processing function
    df = ecg_to_csv(setup)
    
    # Plot in the lay out
    plot_ecg(df,df.columns,csv_name, n_rows = layout[0], n_columns = layout[1], fs = 500, figure_size = (20, 12))
    plt.show()
    print("THE END")