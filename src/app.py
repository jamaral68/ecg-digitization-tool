import streamlit as st
import tempfile
from ultralytics import YOLO
from setup import Setup
from edt import ecg_to_csv

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
layout = (3, 4)
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
        