import streamlit as st

st.title("ECG Digitization Tool")

arquivo = st.file_uploader("Carregar arquivo", type=["png", "jpg", "jpeg", "pdf", "csv"])

if arquivo is not None:
    st.success(f"Arquivo carregado: {arquivo.name}")

    if arquivo.type in ["image/png", "image/jpeg"]:
        st.image(arquivo)