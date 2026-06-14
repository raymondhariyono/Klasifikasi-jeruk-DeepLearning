from src.services.model_service import load_trained_model
from src.ui.sidebar import render_sidebar
from src.ui.pages import (
    render_prediction_page,
    render_project_info_page,
)

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Klasifikasi Kematangan Jeruk Songkit",
        page_icon="🍊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    model = load_trained_model()

    render_sidebar(model_available=model is not None)

    st.title("🍊 Dashboard Klasifikasi Kematangan Jeruk Songkit")
    st.write(
        "Dashboard ini digunakan untuk melakukan prediksi kematangan jeruk songkit "
        "menggunakan model Deep Learning MobileNetV3Large."
    )

    tab_prediction, tab_info = st.tabs([
        "Prediksi & Grad-CAM",
        "Informasi Proyek",
    ])

    with tab_prediction:
        render_prediction_page(model)

    with tab_info:
        render_project_info_page()


if __name__ == "__main__":
    main()