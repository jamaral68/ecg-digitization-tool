import matplotlib.pyplot as plt
from setup import Setup
from edt_utils import plot_ecg
from edt import ecg_to_csv
from ultralytics import YOLO

if __name__ == "__main__":
    image = '../teste.png'               # Path to the input ECG image file to be processed
    csv_name = '../ecg_test2.csv'       # Path and filename for saving the extracted ECG data as a CSV
    pulse_width_mm = 500                 # Horizontal reference: width of one calibration pulse in millimeters (mm)
    mmpsec = 25                          # Paper speed: millimeters per second (mm/s), standard ECG scale
    pulse_per_sec = pulse_width_mm / mmpsec  # Duration of one calibration pulse in seconds
    sample_frequency = 500               # ECG signal sampling frequency in Hertz (Hz)
    time_lead = 2.5                      # Duration of each ECG lead segment in seconds
    num_sampling_points = int(time_lead * sample_frequency)  # Number of samples per lead segment
    layout = (3, 4)                      # Layout of the ECG plot grid: (rows, columns)

    setup = Setup(
        image=image,
        csv_name=csv_name,
        pulse_per_sec=pulse_per_sec,
        sample_frequency=sample_frequency,
        num_sampling_points=num_sampling_points
    )

    model = YOLO("best.pt")
    df = ecg_to_csv(setup, model, save_overlay=True)

    lead_order = ['I','aVR','V1','V4',
                  'II','aVL','V2','V5',
                  'III','aVF','V3','V6']

    fig = plot_ecg(df, lead_order, title="ECG", n_rows=layout[0], n_columns=layout[1], fs=setup.sample_frequency)
    plt.show()
    df.to_csv(setup.csv_name, index=False)
    print("THE END")