import streamlit as st
import tempfile
import cv2 as cv
from ultralytics import YOLO
from setup import Setup
from edt import ecg_to_csv_yolo
from edt_utils import plot_ecg, create_zip

st.title("ECG DIGITIZATION TOOL", anchor=False)

col1, col2, col3, col4 = st.columns(4)

with col1:
    pulse_width_mm = st.number_input('Pulse width (per mm)', value=500)

with col2:
    mm_per_sec = st.number_input('Millimeters per second', value=25)

with col3:
    sample_frequency = st.number_input('Sampling frequency', value=500)

with col4:
    lead_time = st.number_input('Lead duration (seconds)', value=2.5)

pulse_per_sec = pulse_width_mm / mm_per_sec
num_sampling_points = int(lead_time * sample_frequency)

lead_order = [
    'I', 'aVR', 'V1', 'V4',
    'II', 'aVL', 'V2', 'V5',
    'III', 'aVF', 'V3', 'V6',
]

model = YOLO("best.pt")
label_model = YOLO("labels.pt")

uploaded_file = st.file_uploader(
    "Upload file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    if st.button("Scan"):

        csv_name = uploaded_file.name.rsplit(".", 1)[0] + ".csv"

        setup = Setup(
            image=file_path,
            csv_name=csv_name,
            pulse_per_sec=pulse_per_sec,
            sample_frequency=sample_frequency,
            num_sampling_points=num_sampling_points,
        )

        df = ecg_to_csv_yolo(setup, model, label_model=label_model, save_overlay=True)

        if df is not None:
            st.success("Processing completed!")

            fig = plot_ecg(
                df=df,
                columns=lead_order,
                title="Digitized ECG - " + uploaded_file.name,
                n_rows=3,
                n_columns=4,
                fs=sample_frequency
            )

            st.pyplot(fig)

            overlay_path = setup.csv_name.replace(".csv", "_overlay.png")

            with open(overlay_path, "rb") as f:
                overlay_bytes = f.read()

            results = model(file_path)[0]
            yolo_img = results.plot()
            _, yolo_buffer = cv.imencode(".png", yolo_img)
            yolo_bytes = yolo_buffer.tobytes()

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

            zip_file = create_zip(
                csv_bytes=csv_bytes,
                overlay_img=overlay_bytes,
                yolo_img=yolo_bytes,
                csv_name=csv_name
            )

            st.download_button(
                label="Download (CSV + images)",
                data=zip_file,
                file_name="ecg_results.zip",
                mime="application/zip"
            )