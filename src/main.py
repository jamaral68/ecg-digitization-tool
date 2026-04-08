import matplotlib.pyplot as plt
from setup import Setup
from edt_utils import plot_ecg
from edt import ecg_to_csv
from ultralytics import YOLO
 
if __name__ == "__main__":
    image            = '../teste.png'
    csv_name         = '../ecg_test2.csv'
    pulse_width_mm   = 500
    mmpsec           = 25
    pulse_per_sec    = pulse_width_mm / mmpsec
    sample_frequency = 500
    time_lead        = 2.5
    num_sampling_points = int(time_lead * sample_frequency)
    layout           = (3, 4)
 
    setup = Setup(
        image=image,
        csv_name=csv_name,
        pulse_per_sec=pulse_per_sec,
        sample_frequency=sample_frequency,
        num_sampling_points=num_sampling_points,
    )
 
    model       = YOLO("best.pt")
    label_model = YOLO("labels.pt")   # model trained to detect text labels
                                       # set to None to skip inpainting
 
    df = ecg_to_csv(setup, model, label_model=label_model, save_overlay=True)
 
    lead_order = [
        'I',   'aVR', 'V1', 'V4',
        'II',  'aVL', 'V2', 'V5',
        'III', 'aVF', 'V3', 'V6',
    ]
 
    fig = plot_ecg(
        df, lead_order, title="ECG",
        n_rows=layout[0], n_columns=layout[1],
        fs=setup.sample_frequency,
    )
    plt.show()
 
    df.to_csv(setup.csv_name, index=False)
    print("THE END")