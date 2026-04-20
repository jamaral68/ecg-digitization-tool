import streamlit as st
import tempfile
import cv2 as cv
import torch
import os
from ultralytics import YOLO

from setup import Setup
from edt import ecg_to_csv_yolo, ecg_to_csv_cnn
from edt_utils import plot_ecg, create_zip, get_model, predict_and_draw


st.title("ECG DIGITIZATION TOOL", anchor=False)

# =========================
# MODEL SELECTION
# =========================
model_choice = st.selectbox(
    "Select detection model",
    ["YOLO", "Faster R-CNN"]
)

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

# =========================
# LOAD MODELS
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_choice == "YOLO":
    model = YOLO("best.pt")
    label_model = YOLO("labels.pt")

else:
    model = get_model(14)
    model.load_state_dict(torch.load("CNN-leads.pth", map_location=device))
    model.to(device)

    label_model = get_model(2)
    label_model.load_state_dict(torch.load("CNN-labels.pth", map_location=device))
    label_model.to(device)

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Upload file",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded ECG")

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

        # =========================
        # PIPELINE
        # =========================
        if model_choice == "YOLO":

            df = ecg_to_csv_yolo(
                setup,
                model,
                label_model=label_model,
                save_overlay=True
            )

            results = model(file_path)[0]
            yolo_img = results.plot()

            _, buffer = cv.imencode(".png", yolo_img)
            preview_bytes = buffer.tobytes()

        else:

            df = ecg_to_csv_cnn(
                setup,
                model_leads=model,
                device=device,
                label_model=label_model,
                save_overlay=True
            )

            img = cv.imread(file_path)

            preview = predict_and_draw(
                model,
                img,
                device,
                threshold=0.5
            )

            _, buffer = cv.imencode(".png", preview)
            preview_bytes = buffer.tobytes()

        # =========================
        # LOAD GENERATED IMAGES
        # =========================
        overlay_path = setup.csv_name.replace(".csv", "_overlay.png")
        boxes_path = setup.csv_name.replace(".csv", "_boxes.png")

        overlay_bytes = None
        boxes_bytes = None

        if os.path.exists(overlay_path):
            with open(overlay_path, "rb") as f:
                overlay_bytes = f.read()

        if os.path.exists(boxes_path):
            with open(boxes_path, "rb") as f:
                boxes_bytes = f.read()

        # =========================
        # CSV EXPORT
        # =========================
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

        # =========================
        # PLOT ECG
        # =========================
        fig = plot_ecg(
            df=df,
            columns=lead_order,
            title="Digitized ECG - " + uploaded_file.name,
            n_rows=3,
            n_columns=4,
            fs=sample_frequency
        )

        st.pyplot(fig)

        # =========================
        # CREATE ZIP (MODEL-AWARE)
        # =========================
        if model_choice == "YOLO":
            zip_file = create_zip(
                csv_bytes=csv_bytes,
                overlay_img=overlay_bytes,
                boxes_img=preview_bytes,
                csv_name=csv_name
            )

        else:
            zip_file = create_zip(
                csv_bytes=csv_bytes,
                overlay_img=overlay_bytes,
                boxes_img=boxes_bytes,
                csv_name=csv_name
            )

        # =========================
        # DOWNLOAD
        # =========================
        st.download_button(
            label="Download (CSV + images)",
            data=zip_file,
            file_name="ecg_results.zip",
            mime="application/zip"
        )