import streamlit as st
from ultralytics import YOLO

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
    st.write("Arquivo carregado:", arquivo.name)
    st.image(arquivo)
    if st.button('Scan'):
        print('Button clicked')