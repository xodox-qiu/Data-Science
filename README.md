# 📘 Judul Proyek
Mengatasi Curse of Dimensionality pada Prediksi Toksisitas Senyawa Kimia: Pendekatan Seleksi Fitur dan Deep Learning

## 👤 Informasi
- **Nama:** Frezy Ananta Diva Tertiya  
- **Repo:** https://github.com/xodox-qiu/Data-Science
- **Video:** [Link Video Penjelasan Proyek Anda]  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan klasifikasi toksisitas pada dataset dimensi tinggi (*high-dimensional*) dengan sampel terbatas (*small data*).
- Melakukan data preparation meliputi cleaning, encoding, scaling, dan **Feature Selection**.
- Membangun 3 model: **Baseline (Logistic Regression)**, **Advanced (Random Forest)**, **Deep Learning (MLP)**.
- Melakukan evaluasi menggunakan metrik **F1-Score (Weighted)** untuk menangani ketimpangan kelas (*imbalanced data*).

---

# 2. 📄 Problem & Goals
**Problem Statements:** - Dataset memiliki dimensi yang sangat tinggi (1203 fitur) dibanding jumlah sampel yang sedikit (171 baris), berpotensi menyebabkan *overfitting* (Curse of Dimensionality).
- Terdapat ketidakseimbangan kelas (*imbalanced*) antara senyawa Toxic dan Non-Toxic.
- Diperlukan metode untuk memilih fitur kimia yang paling relevan agar komputasi efisien.

**Goals:** - Membangun model yang mampu memprediksi toksisitas dengan F1-Score > 0.60.
- Mereduksi dimensi fitur dari 1203 menjadi 30 fitur paling signifikan menggunakan statistik ANOVA.
- Membandingkan performa model linear, ensemble, dan neural network pada data terbatas.

---
## 📁 Struktur Folder
```
project/
│
├── data/
│
├── notebooks/
│   └── ML_Project_234311040.ipynb
│
├── src/
│   
├── models/
│   ├── model_advanced.pkl
│   ├── model_baseline.pkl
│   └── model_toxicity_dl.keras
│
├── images/
│   └── viz1_class_distribution.png
│   └── viz2_correlation.png
│   └── viz3_dl_loss.png
│   └── viz4_model_comparison.png
│
├── Laporan Proyek Machine Learning.md
├── Checklist Submit.md
├── requirements.txt
├── .gitignore
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository (ID: 728 - Toxicity)
- **Jumlah Data:** 171 Baris, 1203 Fitur
- **Tipe:** Tabular (Quantitative Structure-Activity Relationships / QSAR data)

### Fitur Utama
| Fitur | Deskripsi |
|------|-----------|
| Molecular Descriptors (X1-X1203) | Representasi matematis struktur kimia (Float) |
| Target (Class) | Label klasifikasi: 'Toxic' (1) atau 'Non-Toxic' (0) |

---

# 4. 🔧 Data Preparation
- **Cleaning:** Pengecekan missing values (imputasi mean jika ada) dan Label Encoding untuk target.
- **Feature Selection:** Menggunakan `SelectKBest` (ANOVA F-value) untuk mengambil **30 fitur terbaik** dari 1203 fitur.
- **Transformasi:** Scaling menggunakan `StandardScaler` (Z-score normalization).
- **Splitting:** Stratified Split (70% Train, 15% Val, 15% Test).
- **Handling Imbalance:** Menggunakan `class_weight='balanced'`.

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** **Logistic Regression** (Linear model, weighted classes).
- **Model 2 – Advanced ML:** **Random Forest Classifier** (Ensemble Bagging, max_depth=10, weighted).
- **Model 3 – Deep Learning:** **Multilayer Perceptron (MLP)** dengan arsitektur: Input(30) -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(32, ReLU) -> Dropout(0.3) -> Output(1, Sigmoid).

---

# 6. 🧪 Evaluation
**Metrik:** **F1-Score (Weighted)** & Accuracy. (Dipilih F1-Weighted karena dataset imbalanced).

### Hasil Singkat
| Model | Score (F1-W) | Catatan |
|-------|--------|---------|
| Baseline (LogReg) | 0.62 | Cepat, interpretability tinggi |
| Advanced (RF) | 0.48 | Tangguh terhadap noise, cenderung bias ke mayoritas pada data sangat kecil |
| Deep Learning (MLP) | 0.65 | Memerlukan tuning dropout untuk mencegah overfitting |

---

# 7. 🏁 Kesimpulan
- **Model terbaik:** Deep Learning (Multilayer Perceptron).
- **Alasan:** Memberikan keseimbangan terbaik antara accuracy dan f1-score.
- **Insight penting:** Seleksi fitur dari 1203 menjadi 30 sangat krusial. Tanpa reduksi dimensi, model mengalami *overfitting* parah. Teknik *Class Weighting* membantu model mendeteksi kelas Toxic meskipun jumlah datanya sedikit.

---

# 8. 🔮 Future Work
- [x] Tambah data (Sangat direkomendasikan untuk stabilitas model)
- [x] Feature engineering lebih lanjut (Domain knowledge kimia)
- [x] Tuning model (GridSearch/Bayesian Optimization)
- [x] Coba arsitektur DL lain (Graph Neural Networks)
- [ ] Deployment (API/Web App)

---

# 9. 🔁 Reproducibility
Gunakan environment:
**Python 3.10+**
Libraries utama:
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow` (Keras)
- `ucimlrepo`
- `matplotlib`, `seaborn`
- `joblib`

Instalasi:
```bash
pip install -r requirements.txt
