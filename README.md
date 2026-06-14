# Klasifikasi Kematangan Jeruk Menggunakan SVM RGB+HSV dan MobileNetV3Large

Proyek ini merupakan pengembangan UAS Machine Learning untuk klasifikasi kematangan buah jeruk berdasarkan citra digital. Dataset terdiri dari dua kelas, yaitu `belum_matang` dan `matang`. Proyek ini membandingkan dua pendekatan, yaitu Machine Learning menggunakan SVM berbasis fitur warna RGB+HSV dan Deep Learning menggunakan MobileNetV3Large berbasis transfer learning.

Selain notebook training dan evaluasi model, proyek ini juga menyediakan dashboard berbasis Streamlit untuk melakukan prediksi gambar jeruk baru. Dashboard menampilkan hasil klasifikasi serta interpretasi model menggunakan Grad-CAM agar area citra yang berkontribusi terhadap prediksi dapat diamati secara visual.

## Tujuan Proyek

1. Mengklasifikasikan citra jeruk ke dalam kelas `belum_matang` dan `matang`.
2. Menerapkan metode SVM RGB+HSV sebagai pendekatan Machine Learning.
3. Mengembangkan model Deep Learning menggunakan MobileNetV3Large berbasis transfer learning.
4. Membandingkan performa SVM RGB+HSV dan MobileNetV3Large berdasarkan accuracy, precision, recall, F1-score, dan confusion matrix.
5. Menyediakan dashboard sederhana untuk melakukan prediksi gambar baru.
6. Menampilkan interpretasi model CNN menggunakan Grad-CAM.

## Dataset

Dataset yang digunakan adalah citra jeruk dengan dua kelas.

| Kelas | Jumlah Citra |
|---|---:|
| Belum Matang | 235 |
| Matang | 245 |
| Total | 480 |

Dataset disusun dalam folder kelas seperti berikut.

```text
dataset/
├── belum_matang/
│   ├── belum_matang_001.jpg
│   ├── belum_matang_002.jpg
│   └── ...
└── matang/
    ├── matang_001.jpg
    ├── matang_002.jpg
    └── ...
```

## Metode yang Digunakan

### 1. SVM RGB+HSV

Metode SVM menggunakan fitur warna hasil ekstraksi manual. Setiap citra diubah menjadi enam fitur numerik, yaitu:

- Mean R
- Mean G
- Mean B
- Mean H
- Mean S
- Mean V

Alur SVM RGB+HSV:

1. Membaca citra jeruk.
2. Resize atau crop gambar.
3. Konversi citra ke RGB.
4. Ekstraksi nilai Mean R, Mean G, dan Mean B.
5. Konversi citra ke HSV.
6. Ekstraksi nilai Mean H, Mean S, dan Mean V.
7. Penggabungan fitur RGB+HSV.
8. Standardisasi fitur.
9. Training dan testing model SVM.
10. Evaluasi model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

### 2. MobileNetV3Large

Metode Deep Learning menggunakan MobileNetV3Large berbasis transfer learning. Model menerima input citra RGB berukuran 224×224×3 dan mempelajari fitur visual secara otomatis dari gambar.

Alur MobileNetV3Large:

1. Membaca dataset citra jeruk.
2. Resize gambar ke ukuran 224×224.
3. Pembagian data menjadi training, validation, dan testing.
4. Data augmentation pada data training.
5. Training MobileNetV3Large berbasis transfer learning.
6. Evaluasi model.
7. Visualisasi accuracy, loss, confusion matrix, dan Grad-CAM.
8. Demo prediksi gambar baru melalui dashboard Streamlit.

## Struktur Folder Proyek

Struktur folder dapat disesuaikan, tetapi susunan yang disarankan adalah sebagai berikut.

```text
UAS_Machine_Learning_Jeruk/
├── dataset/
│   ├── belum_matang/
│   └── matang/
├── notebook/
│   ├── svm_rgb_hsv_jeruk_modelUTS.ipynb
│   └── train_mobilenetv3_jeruk.ipynb
├── modelUTS/
│   ├── best_svm_model.pkl
│   ├── scaler.pkl
│   └── reports/
│       ├── test_metrics_best_svm.json
│       ├── classification_report_best_svm.csv
│       └── confusion_matrix_best_svm.png
├── model_MobileNetv3Large/
│   └── mobilenetv3_orange_maturity.keras
├── dashboard_streamlit/
│   └── orange_maturity_dashboard/
│       ├── app.py
│       ├── requirements.txt
│       ├── models/
│       │   └── mobilenetv3_orange_maturity.keras
│       └── src/
│           ├── services/
│           └── utils/
├── reports/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── model_comparison_metrics.png
│   ├── dataset_distribution.png
│   └── prediction_summary_correct_wrong.png
├── laporan_akhir.pdf
├── presentasi_uas.pptx
└── README.md
```

## Library yang Digunakan

Library utama yang digunakan dalam proyek ini adalah:

```text
tensorflow
keras
numpy
pandas
matplotlib
scikit-learn
opencv-python
Pillow
joblib
streamlit
```

## Cara Menjalankan Notebook Training dan Evaluasi

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

### 2. Install Library

Jika dijalankan secara lokal, install library yang dibutuhkan:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn opencv-python pillow joblib streamlit
```

Jika menggunakan Google Colab, sebagian besar library sudah tersedia. Library tambahan dapat dipasang dengan:

```python
!pip install opencv-python joblib streamlit
```

### 3. Jalankan Notebook SVM RGB+HSV

Buka notebook berikut:

```text
notebook/svm_rgb_hsv_jeruk_modelUTS.ipynb
```

Jalankan seluruh cell dari awal sampai akhir. Notebook ini akan melakukan load dataset, ekstraksi fitur RGB+HSV, standardisasi fitur, training beberapa skenario SVM, evaluasi model terbaik, dan penyimpanan hasil model serta laporan evaluasi.

### 4. Jalankan Notebook MobileNetV3Large

Buka notebook berikut:

```text
notebook/train_mobilenetv3_jeruk.ipynb
```

Jalankan seluruh cell dari awal sampai akhir. Notebook ini akan melakukan load dataset, split data, preprocessing citra, data augmentation, training MobileNetV3Large, evaluasi model, penyimpanan grafik hasil, confusion matrix, Grad-CAM, dan demo prediksi gambar baru.

## Cara Menjalankan Dashboard Streamlit

Dashboard Streamlit berada pada folder:

```text
dashboard_streamlit/orange_maturity_dashboard/
```

File utama dashboard:

```text
dashboard_streamlit/orange_maturity_dashboard/app.py
```

Model terlatih yang digunakan dashboard:

```text
dashboard_streamlit/orange_maturity_dashboard/models/mobilenetv3_orange_maturity.keras
```

Jika file model belum ada pada folder `models/`, salin model dari:

```text
model_MobileNetv3Large/mobilenetv3_orange_maturity.keras
```

### Langkah Menjalankan Dashboard di Windows

Jalankan perintah berikut dari root proyek:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r dashboard_streamlit/orange_maturity_dashboard/requirements.txt
cd dashboard_streamlit/orange_maturity_dashboard
streamlit run app.py
```

Setelah perintah dijalankan, Streamlit akan menampilkan URL lokal, biasanya seperti berikut:

```text
http://localhost:8501
```

Buka URL tersebut melalui browser untuk menggunakan dashboard.

### Langkah Menjalankan Dashboard di Linux atau macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r dashboard_streamlit/orange_maturity_dashboard/requirements.txt
cd dashboard_streamlit/orange_maturity_dashboard
streamlit run app.py
```

## Cara Menggunakan Dashboard

1. Buka dashboard melalui browser.
2. Unggah gambar jeruk dalam format JPG, JPEG, atau PNG.
3. Dashboard akan melakukan preprocessing gambar ke ukuran RGB 224×224.
4. Model MobileNetV3Large akan memprediksi kelas gambar.
5. Dashboard menampilkan hasil prediksi, confidence, dan visualisasi Grad-CAM.

Output prediksi terdiri dari dua kelas:

```text
Belum Matang
Matang
```

## Contoh Input dan Output Dashboard

Contoh input:

```text
Gambar jeruk baru dalam format JPG/JPEG/PNG
```

Contoh output:

```text
Prediksi: Belum Matang
Confidence: 100.00%
```

Dashboard juga menampilkan Grad-CAM, yaitu heatmap yang ditumpangkan pada gambar input. Area dengan warna lebih kuat menunjukkan bagian citra yang lebih berkontribusi terhadap keputusan model. Misalnya, area pada kulit jeruk yang masih hijau dapat menjadi bagian penting ketika model memprediksi kelas `Belum Matang`.

Catatan: nilai confidence yang tinggi tidak selalu menjamin prediksi sempurna. Hasil prediksi tetap perlu diperiksa secara visual, terutama jika gambar memiliki pencahayaan ekstrem, bayangan kuat, background ramai, atau warna jeruk yang ambigu.

## Hasil Utama Model

### Hasil SVM RGB+HSV

Model SVM terbaik diperoleh pada skenario berikut.

| Parameter | Nilai |
|---|---|
| Kernel | RBF |
| C | 100 |
| Gamma | scale |

Hasil evaluasi SVM RGB+HSV:

| Metrik | Nilai |
|---|---:|
| Testing Accuracy | 90,62% |
| Precision Macro | 91,14% |
| Recall Macro | 90,51% |
| F1-score Macro | 90,57% |

Confusion matrix SVM RGB+HSV:

| Aktual / Prediksi | Belum Matang | Matang |
|---|---:|---:|
| Belum Matang | 40 | 7 |
| Matang | 2 | 47 |

### Hasil MobileNetV3Large

Model MobileNetV3Large menggunakan pendekatan transfer learning dengan input citra RGB berukuran 224×224×3.

Hasil training MobileNetV3Large:

| Metrik | Nilai |
|---|---:|
| Training Accuracy | 99,40% |
| Validation Accuracy | 98,61% |
| Training Loss | 0,0228 |
| Validation Loss | 0,0486 |
| Epoch Aktual | 137 epoch |
| Batch Size | 16 |

Hasil testing MobileNetV3Large:

| Metrik | Nilai |
|---|---:|
| Testing Accuracy | 98,61% |
| Precision | 98,65% |
| Recall | 98,61% |
| F1-score | 98,61% |

Confusion matrix MobileNetV3Large:

| Aktual / Prediksi | Belum Matang | Matang |
|---|---:|---:|
| Belum Matang | 35 | 1 |
| Matang | 0 | 36 |

## Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| SVM RGB+HSV | 90,62% | 91,14% | 90,51% | 90,57% |
| MobileNetV3Large | 98,61% | 98,65% | 98,61% | 98,61% |

Berdasarkan hasil evaluasi, MobileNetV3Large memperoleh performa terbaik pada seluruh metrik evaluasi. Model ini lebih unggul karena mampu mempelajari fitur visual secara otomatis dari citra, seperti warna, tekstur, bentuk, bayangan, dan pola permukaan jeruk. Sementara itu, SVM RGB+HSV hanya menggunakan enam fitur warna hasil ekstraksi manual.

## Visualisasi Hasil

Visualisasi hasil yang digunakan dalam proyek ini meliputi:

1. Grafik training accuracy dan validation accuracy.
2. Grafik training loss dan validation loss.
3. Confusion matrix SVM RGB+HSV.
4. Confusion matrix MobileNetV3Large.
5. Grafik perbandingan performa model.
6. Grafik distribusi dataset.
7. Grafik jumlah prediksi benar dan salah.
8. Contoh hasil klasifikasi benar dan salah.
9. Grad-CAM untuk interpretasi area citra yang diperhatikan model.

## Kesimpulan

Proyek ini menunjukkan bahwa klasifikasi kematangan jeruk dapat dilakukan menggunakan pendekatan Machine Learning dan Deep Learning. SVM RGB+HSV mampu memberikan performa yang cukup baik dengan accuracy sebesar 90,62%. Namun, MobileNetV3Large memberikan hasil lebih tinggi dengan accuracy sebesar 98,61%.

Berdasarkan hasil tersebut, MobileNetV3Large menjadi model terbaik dalam proyek ini karena memiliki performa lebih tinggi pada accuracy, precision, recall, dan F1-score. Pendekatan Deep Learning berbasis transfer learning lebih efektif dibandingkan SVM RGB+HSV karena mampu mempelajari fitur visual citra secara otomatis.

## Link GitHub

```text
https://github.com/raymondhariyono/Klasifikasi-jeruk-DeepLearning
```
