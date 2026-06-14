import streamlit as st
import tensorflow as tf

from src.config.settings import MODEL_PATH


@st.cache_resource(show_spinner="Memuat model...")
def load_trained_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    except Exception as error:
        st.error("Model gagal dimuat.")
        st.code(str(error))
        return None