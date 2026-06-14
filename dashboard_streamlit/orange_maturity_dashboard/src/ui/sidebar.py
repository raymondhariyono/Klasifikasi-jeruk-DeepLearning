import streamlit as st


def render_sidebar(model_available: bool) -> None:
    st.sidebar.header("Informasi Model")

    st.sidebar.write("**Model:** MobileNetV3Large")
    st.sidebar.write("**Metode:** Transfer Learning")
    st.sidebar.write("**Kelas utama:** Belum Matang dan Matang")
    st.sidebar.write("**Input:** RGB 224×224")
    st.sidebar.write("**Output:** Sigmoid binary classification")

    st.sidebar.divider()

    if model_available:
        st.sidebar.success("Model berhasil dimuat.")
    else:
        st.sidebar.error("Model belum tersedia.")

    st.sidebar.caption(
        "Kategori transisi dan matang sedikit kehijauan merupakan interpretasi "
        "tambahan berdasarkan probabilitas, bukan label training baru."
    )