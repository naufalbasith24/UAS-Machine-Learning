1) Tujuan & algoritma (beserta alasannya)

Tujuan utama:

Regresi: memprediksi besarnya area terbakar (area). Karena distribusinya sangat skewed dan banyak nol, target dipakai log1p(area) agar error lebih stabil dan tidak didominasi outlier.

Klasifikasi: memprediksi apakah terjadi kebakaran sama sekali (fire_occurred = area > 0). Ini menjawab pertanyaan biner: ada/tidak ada area terbakar.

Algoritma yang digunakan & alasan:

Random Forest Regressor (untuk log1p(area)): kuat untuk relasi non-linear, tahan terhadap outlier, bekerja baik pada campuran fitur numerik & kategorikal, dan minim tuning.

Random Forest Classifier (untuk fire_occurred): alasan sama seperti di atas; juga bagus untuk ketidakseimbangan moderat karena sifat bagging.

Pra-proses:

Fitur numerik → StandardScaler.

Fitur kategorikal (month, day) → OneHotEncoder.

Evaluasi:

Regresi → RMSE (skala log), R², dan baseline RMSE (prediksi rata-rata) untuk pembanding.

Klasifikasi → Accuracy, Precision, Recall, F1, serta Confusion Matrix.
