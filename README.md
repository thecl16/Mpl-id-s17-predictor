# ⚔️ MPL ID S17 — Machine Learning Match Predictor

Proyek end-to-end untuk memprediksi hasil pertandingan MPL ID Season 17 Playoffs  
menggunakan XGBoost + Streamlit.

---

## 📁 Struktur Proyek

```
mpl_predictor/
│
├── data/                        ← Dibuat otomatis saat scraping
│   ├── mpl_raw_data.csv         ← Hasil scraping mentah
│   ├── dataset_siap_ml.csv      ← Dataset dengan fitur ML
│   └── team_encoder.json        ← Mapping nama tim → angka
│
├── models/                      ← Dibuat otomatis saat training
│   ├── model_mpl_s17.pkl        ← Model XGBoost tersimpan
│   ├── training_report.json     ← Metrik evaluasi
│   ├── confusion_matrix.png     ← Plot confusion matrix
│   └── feature_importance.png   ← Plot feature importance
│
├── scripts/
│   ├── 1_scrape_data.py         ← Bagian 1: Scraping
│   ├── 2_feature_engineering.py ← Bagian 2: Feature Engineering
│   ├── 3_train_model.py         ← Bagian 3: Training Model
│   └── 4_app.py                 ← Bagian 4: Streamlit App
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

### Langkah 0 — Persiapan Environment

```bash
# Clone / extract project, lalu masuk ke folder
cd mpl_predictor

# (Opsional tapi disarankan) buat virtual environment
python -m venv venv

# Aktifkan venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install semua dependensi
pip install -r requirements.txt
```

---

### Langkah 1 — Scraping Data

```bash
python scripts/1_scrape_data.py
```

**Output:** `data/mpl_raw_data.csv`

> ℹ️ Jika Liquipedia tidak bisa diakses (firewall/network), skrip otomatis  
> menggunakan **synthetic fallback data** yang realistis agar pipeline tetap berjalan.

---

### Langkah 2 — Feature Engineering

```bash
python scripts/2_feature_engineering.py
```

**Output:**
- `data/dataset_siap_ml.csv` — dataset dengan fitur lengkap
- `data/team_encoder.json` — encoder nama tim

**Fitur yang dibuat:**
| Fitur | Keterangan |
|-------|-----------|
| `rolling_win_rate` | Win rate 5 pertandingan terakhir |
| `h2h_win_rate` | Win rate head-to-head antar tim |
| `avg_score_diff` | Rata-rata selisih skor |
| `win_streak` | Streak kemenangan/kekalahan |
| `team_encoded` | Nama tim dikodekan menjadi angka |
| `diff_*` | Selisih fitur antara Team A dan B |

---

### Langkah 3 — Training Model

```bash
python scripts/3_train_model.py
```

**Output:**
- `models/model_mpl_s17.pkl` — model XGBoost tersimpan
- `models/training_report.json` — akurasi, AUC, CV score
- `models/confusion_matrix.png` — visualisasi evaluasi
- `models/feature_importance.png` — visualisasi fitur penting

**Metrik yang ditampilkan:**
- Accuracy Score (Test & Cross-Validation)
- ROC-AUC Score
- Classification Report (Precision, Recall, F1)
- Confusion Matrix

---

### Langkah 4 — Jalankan Aplikasi Web

```bash
streamlit run scripts/4_app.py
```

Buka browser ke: **http://localhost:8501**

**Fitur aplikasi:**
- 🔵🔴 Dropdown pilih Team A vs Team B
- ⚡ Tombol **PREDICT OUTCOME**
- 🏆 Prediksi pemenang + persentase keyakinan
- 📊 Grafik batang probabilitas interaktif (Plotly)
- 🔍 Breakdown statistik tiap tim

---

## ⚠️ Catatan Penting

1. **Liquipedia Rate Limit** — Skrip sudah menambahkan delay 2 detik antar request.  
   Jangan dijalankan terlalu sering untuk menghormati server.

2. **Data Aktualitas** — Data S17 baru tersedia setelah semua pertandingan selesai.  
   Semakin banyak data, semakin akurat model.

3. **Synthetic Data** — Jika scraping gagal, fallback data sudah dikalibrasi  
   berdasarkan pola nyata MPL ID (team strength, format best-of-3/5).

4. **Model Accuracy** — Dengan dataset terbatas (~100 matches), ekspektasi akurasi  
   berkisar 60–75%. Akurasi meningkat dengan lebih banyak data historis.

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| `FileNotFoundError: mpl_raw_data.csv` | Jalankan script 1 terlebih dahulu |
| `FileNotFoundError: model_mpl_s17.pkl` | Jalankan script 1, 2, 3 secara berurutan |
| Streamlit port sudah dipakai | `streamlit run scripts/4_app.py --server.port 8502` |
| Scraping return 0 data | Normal — fallback data otomatis digunakan |

---

## 📚 Tech Stack

- **Scraping:** `requests` + `BeautifulSoup4`
- **Data:** `pandas` + `numpy`
- **ML:** `scikit-learn` + `XGBoost`
- **Visualization:** `matplotlib` + `seaborn` + `plotly`
- **Web App:** `Streamlit`
