import matplotlib.pyplot as plt
from setup import Setup
from edt_utils import plot_ecg
from edt import ecg_to_csv

if __name__ == "__main__":

    image = '../teste.png'        # Path to the ECG image used as input
    csv_name = '../ecg_test2.csv' # Path where the output CSV will be saved
    pulse_width_mm = 5            # Width of reference pulse (used to compute time scale)
    pulse_height_mm = 10          # Height of reference pulse (used to compute voltage scale)
    mmpsec = 25                   # mm per second (standard ECG paper speed)
    mmpmv = 10                    # mm per mV (standard ECG amplitude scaling)
    pulse_per_sec = pulse_width_mm / mmpsec   # Conversion factor: pixels → seconds (used in convert_to_secmv)
    pulse_per_mv = pulse_height_mm / mmpmv    # Conversion factor: pixels → mV (used in convert_to_secmv)
    sample_frequency = 500        # Sampling frequency in Hz (used for plotting time axis)
    time_lead = 2.5              # Duration of each ECG lead in seconds

    # Total number of samples per lead after interpolation (defines signal resolution)
    num_sampling_points = int(time_lead * sample_frequency) 

    # ECG display layout (rows x columns) used to arrange leads in the plot grid
    layout = (3, 4)             

    setup = Setup(
        image=image,                         
        csv_name=csv_name,                   
        pulse_per_sec=pulse_per_sec,         
        pulse_per_mv=pulse_per_mv,           
        sample_frequency=sample_frequency,   
        num_sampling_points=num_sampling_points  
    )

    df = ecg_to_csv(setup)  

    lead_order = [
        'I', 'aVR', 'V1', 'V4',
        'II', 'aVL', 'V2', 'V5',
        'III', 'aVF', 'V3', 'V6'
    ]

    # Plot ECG signals in grid layout
    plot_ecg(
        df,
        lead_order,
        csv_name,                      # Used as figure title
        n_rows=layout[0],              # Number of subplot rows
        n_columns=layout[1],           # Number of subplot columns
        fs=setup.sample_frequency,     # Sampling frequency for time axis
        figure_size=(20, 12)           # Size of the figure
    )

    plt.show()

    # Save extracted ECG signals to CSV
    df.to_csv(setup.csv_name, index=False)

    print("THE END")