1. Tujuan & Algoritma 

a. Tujuan Analisis

Dataset Forest Fires berisi data meteorologi dan kondisi lingkungan pada area hutan di Portugal, dengan target luas area terbakar (area). Tujuan analisis ini dibagi menjadi dua pendekatan:

1. Regresi:
- Target: log1p(area)
- Tujuan: memprediksi besarnya area terbakar (dalam hektar). Fungsi log1p digunakan agar distribusi target yang sangat skewed menjadi lebih normal, sehingga model lebih stabil dan tidak terlalu dipengaruhi oleh outlier (misalnya kebakaran sangat besar).


2. Klasifikasi:
- Target: fire_occurred = 1 jika area > 0, 0 jika tidak terbakar.
- Tujuan: memprediksi kemungkinan terjadinya kebakaran hutan sama sekali.



b. Algoritma yang Digunakan

1. Random Forest Regressor
Alasan: 
- Mampu menangani relasi non-linear yang kompleks antara faktor cuaca/lingkungan dan luas area terbakar.
- Tahan terhadap outlier dan data campuran (numerik + kategorikal).
- Tidak memerlukan asumsi distribusi data yang ketat seperti linear regression.


2. Random Forest Classifier
Alasan:
- Sama seperti di atas, tetapi fokus pada prediksi kelas (ada/tidak kebakaran).
- Bagging dalam Random Forest mengurangi risiko overfitting.

3. Pra-Pemrosesan:
- Fitur numerik: Standarisasi menggunakan StandardScaler.
- Fitur kategorikal (month, day): diubah menjadi one-hot encoding dengan OneHotEncoder.

4. Evaluasi:
- Regresi → RMSE (log-scale), R², baseline RMSE (prediksi rata-rata).
- Klasifikasi → Accuracy, Precision, Recall, F1-score, Confusion Matrix.




2. Analisis Dataset

a. Gambaran Umum Dataset
Berdasarkan hasil EDA:
- Jumlah data: 517 baris
- Jumlah kolom: 13 fitur asli + 2 fitur turunan (log_area, fire_occurred).
- Fitur utama:
    - Lokasi (X, Y)
    - Waktu (month, day)
    - Indeks cuaca kebakaran (FFMC, DMC, DC, ISI)
    - Kondisi cuaca (temp, RH, wind, rain)
    - Target asli: area
- Missing values: Tidak ada
- Distribusi target area: sangat skewed, sebagian besar kebakaran berukuran kecil (≤1 ha), beberapa kasus ekstrem > 100 ha.


b. Hasil Regresi (Prediksi Luas Area Terbakar)
- Model: Random Forest Regressor
- Target: log1p(area)
- Hasil:
    - RMSE (log-scale): 1.3454
    - Baseline RMSE (mean-predictor): 1.9028
    - R²: 0.5283
- Interpretasi:
    - Model mampu menjelaskan sekitar 53% variasi pada target (skala log).
    - RMSE yang jauh lebih kecil dari baseline menunjukkan model lebih baik daripada prediksi rata-rata.


c. Hasil Klasifikasi (Prediksi Ada/Tidaknya Kebakaran)
- Model: Random Forest Classifier
- Target: fire_occurred
- Hasil:
    - Accuracy: 0.9750#
    - Precision: 0.9600
    - Recall: 0.9231
    - F1-score: 0.9412
- Confusion Matrix:
    [[69,  1],
     [ 2, 24]]

- Interpretasi:
    - Model dapat mengklasifikasikan kejadian kebakaran dengan akurasi tinggi.
    - Precision tinggi berarti prediksi kebakaran jarang salah (false positive sedikit).
    - Recall cukup tinggi artinya sebagian besar kejadian kebakaran terdeteksi.

d. Temuan Penting
1. Cuaca (terutama temp, RH, wind, rain) dan indeks kebakaran (FFMC, DMC, DC, ISI) sangat mempengaruhi hasil model.
2. Sebagian besar data menunjukkan kebakaran kecil atau tidak ada kebakaran sama sekali.
3. Regresi lebih sulit dilakukan karena target sangat skewed, namun pendekatan log-transform membantu.
4. Klasifikasi menunjukkan performa sangat baik, cocok digunakan untuk sistem peringatan dini.
