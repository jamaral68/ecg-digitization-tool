import matplotlib.pyplot as plt
from setup import Setup
from edt_utils import plot_ecg
from edt import ecg_to_csv
from ultralytics import YOLO

if __name__ == "__main__":

    image = '../teste.png' # Path to the input ECG image file to be processed
    csv_name = '../ecg_test2.csv' # Path and filename for saving the extracted ECG data as a CSV
    pulse_width_mm = 500       # Horizontal reference 
    pulse_height_mm = 1000   # Vertical reference 
    mmpsec = 25              # Standard paper speed in mm/s 
    mmpmv = 10               # Standard amplitude scale in mm/mV 
    pulse_per_sec = pulse_width_mm / mmpsec   # Fraction of a second represented by the pulse width
    pulse_per_mv = pulse_height_mm / mmpmv    # Voltage per mm in the vertical axis
    sample_frequency = 500                     # ECG signal sampling frequency in Hz
    time_lead = 2.5                            # Duration in seconds of each lead segment
    num_sampling_points = int(time_lead * sample_frequency)  # Number of samples per lead segment
    layout = (3, 4)                            # Number of rows and columns in the plot (3x4 grid)

    # Initialize Setup object with all parameters
    setup = Setup(
        image=image,
        csv_name=csv_name,
        pulse_per_sec=pulse_per_sec,
        pulse_per_mv=pulse_per_mv,
        sample_frequency=sample_frequency,
        num_sampling_points=num_sampling_points
    )

    # Load YOLO model for ECG waveform detection (loaded only once)
    model = YOLO("best.pt")

    # Extract ECG signal from the image and convert to a DataFrame
    df = ecg_to_csv(setup, model=model)

    # Define the standard 12-lead ECG order for plotting
    lead_order = ['I', 'aVR', 'V1', 'V4',
                  'II', 'aVL', 'V2', 'V5',
                  'III', 'aVF', 'V3', 'V6']

    # Plot the ECG using the extracted DataFrame
    fig = plot_ecg(
        df,
        lead_order,
        title="ECG",
        n_rows=layout[0],            # Number of rows in the plot grid
        n_columns=layout[1],         # Number of columns in the plot grid
        fs=setup.sample_frequency,   # Sampling frequency for the x-axis
        figure_size=(20,12)          # Size of the figure in inches
    )

    plt.show()
    df.to_csv(setup.csv_name, index=False)
    print("THE END")