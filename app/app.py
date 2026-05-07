import sys
import tempfile
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecg_digitizer.config import DigitizerConfig  # noqa: E402
from ecg_digitizer.digitizer import ecg_to_csv  # noqa: E402
from ecg_digitizer.utils import plot_ecg_signal  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"

st.title("ECG DIGITIZATION TOOL", anchor=False)

col1, col2, col3, col4 = st.columns(4)

with col1:
    pulse_width_mm = st.number_input("Pulse width (per mm)", value=500)

with col2:
    mm_per_sec = st.number_input("Millimeters per second", value=25)

with col3:
    sample_frequency = st.number_input("Sampling frequency", value=500)

with col4:
    lead_time = st.number_input("Lead duration (seconds)", value=2.5)

pulse_per_sec = pulse_width_mm / mm_per_sec
num_sampling_points = int(lead_time * sample_frequency)

lead_order = [
    "I",
    "aVR",
    "V1",
    "V4",
    "II",
    "aVL",
    "V2",
    "V5",
    "III",
    "aVF",
    "V3",
    "V6",
]


@st.cache_resource
def load_models():
    return YOLO(str(MODELS_DIR / "best.pt")), YOLO(str(MODELS_DIR / "labels.pt"))


model, label_model = load_models()

uploaded_file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.getbuffer())
        file_path = tmp.name

    if st.button("Scan"):
        csv_name = uploaded_file.name.rsplit(".", 1)[0] + ".csv"

        config = DigitizerConfig(
            image=file_path,
            csv_name=csv_name,
            pulse_per_sec=pulse_per_sec,
            sample_frequency=sample_frequency,
            num_sampling_points=num_sampling_points,
        )

        df, lead_crops = ecg_to_csv(config, model, label_model=label_model, save_overlay=True)

        if df is not None and not df.empty:
            st.success("Processing completed!")

            if lead_crops:
                st.subheader("Lead crops")
                n_cols_crops = 3
                n_rows_crops = (len(lead_crops) + n_cols_crops - 1) // n_cols_crops
                fig_crops, ax_crops = plt.subplots(
                    nrows=n_rows_crops, ncols=n_cols_crops, figsize=(14, 3 * n_rows_crops)
                )
                ax_crops = np.atleast_1d(ax_crops).flatten()
                for i, (lead_name, img_lead) in enumerate(lead_crops.items()):
                    ax_crops[i].imshow(img_lead, cmap="gray")
                    ax_crops[i].set_title(f"Lead: {lead_name}", fontsize=13)
                    ax_crops[i].axis("off")
                for j in range(len(lead_crops), len(ax_crops)):
                    fig_crops.delaxes(ax_crops[j])
                fig_crops.tight_layout()
                st.pyplot(fig_crops)

            # Casa o lead_order (uppercase) com as colunas reais do df
            # (digitizer.py emite lead_name.lower()).
            available = {col.lower(): col for col in df.columns}
            ordered_cols = [available[lead.lower()] for lead in lead_order if lead.lower() in available]

            n_cols = 4
            n_rows = (len(ordered_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
            fig.suptitle("Digitized ECG - " + uploaded_file.name, fontsize=18)
            axes = np.atleast_2d(axes).flatten()

            for idx, col in enumerate(ordered_cols):
                series = df[col].dropna()
                ts = np.asarray(series.index, dtype=float)
                plot_ecg_signal(ts, series.values, axes[idx])
                axes[idx].set_title(col.upper(), fontsize=11)

            for idx in range(len(ordered_cols), len(axes)):
                axes[idx].set_visible(False)

            plt.subplots_adjust(top=0.92, hspace=0.5, wspace=0.4)
            st.pyplot(fig)

            csv_bytes = df.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
            )
        else:
            st.error("No leads were digitized.")
