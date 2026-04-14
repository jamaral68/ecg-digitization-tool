import streamlit as st
import tempfile
import matplotlib.pyplot as plt
from ultralytics import YOLO
from setup import Setup
from edt import ecg_to_csv
from edt_utils import plot_ecg

st.title("ECG-DIGITIZATION-TOOL", text_alignment="left")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pulse_width_mm = st.number_input('Pulse per mm width', value=500)

with col2:
    mmpsec = st.number_input('Millimeter per second', value=25)

with col3:
    sample_frequency = st.number_input('Sample frequency', value=500)

with col4:
    time_lead = st.number_input('Time lead', value=2.5)

pulse_per_sec       = pulse_width_mm / mmpsec
num_sampling_points = int(time_lead * sample_frequency)
lead_order = [
    'I',   'aVR', 'V1', 'V4',
    'II',  'aVL', 'V2', 'V5',
    'III', 'aVF', 'V3', 'V6',
]

model = YOLO("best.pt")
label_model = YOLO("labels.pt")

arquivo = st.file_uploader("Carregar arquivo", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if arquivo is not None:
    st.image(arquivo)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(arquivo.getbuffer())
        caminho = tmp.name
    
    if st.button('Scan'):
        name = arquivo.name    
        csv_name = name.rsplit('.', 1)[0] + '.csv'

        setup = Setup(
            image=caminho,
            csv_name=csv_name,
            pulse_per_sec=pulse_per_sec,
            sample_frequency=sample_frequency,
            num_sampling_points=num_sampling_points,
        )

        df = ecg_to_csv(setup, model, label_model=label_model, save_overlay=True)
        
        if df is not None:
            fig = plot_ecg(
                df=df,
                columns=lead_order,
                title="ECG Digitalizado - "+arquivo.name,
                n_rows=3,
                n_columns=4,
                fs=sample_frequency
            )

            st.pyplot(fig)

            st.download_button(
                label="Baixar CSV",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name=csv_name,
                mime="text/csv"
            )