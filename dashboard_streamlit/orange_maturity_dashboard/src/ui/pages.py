from PIL import Image
import streamlit as st

from src.services.prediction_service import predict_image
from src.services.gradcam_service import make_gradcam_heatmap, create_gradcam_overlay


def render_prediction_page(model) -> None:
    st.header("Prediksi & Grad-CAM")

    uploaded_file = st.file_uploader(
        "Upload gambar jeruk",
        type=["jpg", "jpeg", "png"],
        key="prediction_uploader",
    )

    if uploaded_file is None:
        st.info("Upload gambar jeruk untuk melakukan prediksi dan menampilkan Grad-CAM.")
        return

    image = Image.open(uploaded_file)

    if model is None:
        st.error("Model belum tersedia. Pastikan file model berada di folder `models/`.")
        return

    prediction_result = predict_image(model, image)

    col_image, col_metrics, col_gradcam = st.columns([1, 1, 1])

    with col_image:
        st.subheader("Gambar Input")
        st.image(image, width='stretch')

    with col_metrics:
        st.subheader("Hasil Prediksi")
        st.metric("Prediksi Biner", prediction_result["binary_class"])
        st.metric("Interpretasi", prediction_result["interpreted_class"])
        st.metric("Confidence", f"{prediction_result['confidence'] * 100:.2f}%")

        st.write("Status:", prediction_result["confidence_status"])

        st.write("Probabilitas Belum Matang")
        st.progress(float(prediction_result["probability_belum_matang"]))
        st.write(f"{prediction_result['probability_belum_matang'] * 100:.2f}%")

        st.write("Probabilitas Matang")
        st.progress(float(prediction_result["probability_matang"]))
        st.write(f"{prediction_result['probability_matang'] * 100:.2f}%")

        if prediction_result["final_label"] == "transisi":
            st.warning(
                "Gambar berada pada area transisi. Model hanya dilatih dengan dua kelas utama, "
                "sehingga hasil ini ditampilkan sebagai interpretasi tambahan."
            )
        elif prediction_result["final_label"] == "matang":
            st.success("Jeruk cenderung masuk kategori matang.")
        else:
            st.info("Jeruk cenderung masuk kategori belum matang.")

    with col_gradcam:
        st.subheader("Grad-CAM")
        heatmap = make_gradcam_heatmap(model, prediction_result["input_array"])

        if heatmap is None:
            st.error("Grad-CAM gagal dibuat.")
        else:
            original_np, heatmap_resized, overlay = create_gradcam_overlay(image, heatmap)
            st.image(heatmap_resized, width='stretch')
            st.image(overlay, width='stretch')


# report and separate grad-cam page removed; prediction page now includes Grad-CAM


def render_project_info_page() -> None:
    st.header("Informasi Proyek")

    st.subheader("Arsitektur Model")
    st.write(
        "Model menggunakan MobileNetV3Large pretrained ImageNet sebagai feature extractor. "
        "Bagian akhir model terdiri dari GlobalAveragePooling2D, Dropout, dan Dense sigmoid "
        "untuk klasifikasi biner."
    )

    st.subheader("Kelas dan Interpretasi")
    st.write(
        "Model dilatih menggunakan dua kelas utama, yaitu belum matang dan matang. "
        "Kategori seperti transisi, matang sedikit kehijauan, dan matang sempurna merupakan "
        "interpretasi tambahan berdasarkan probabilitas output model, bukan label training baru."
    )

    st.subheader("Batasan")
    st.write(
        "Model dapat dipengaruhi oleh variasi background, pencahayaan, dan tingkat kematangan "
        "yang berada di antara dua kelas utama. Pengembangan berikutnya dapat menambahkan kelas "
        "setengah matang atau menggunakan segmentasi objek jeruk."
    )

    st.subheader("Poin Tambahan")
    st.write("- Data augmentation")
    st.write("- Transfer learning MobileNetV3Large")
    st.write("- Imbalance handling dengan stratified split dan class weight")
    st.write("- Evaluasi accuracy, precision, recall, F1-score")
    st.write("- Grad-CAM untuk interpretasi model")
    st.write("- Dashboard Streamlit untuk demo prediksi")