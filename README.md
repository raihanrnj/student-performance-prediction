# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Meskipun telah mencetak banyak lulusan berkualitas, institusi ini menghadapi tantangan serius berupa **tingginya angka dropout mahasiswa**. Dropout yang tinggi berdampak pada penurunan reputasi institusi, kerugian finansial, dan dampak sosial bagi mahasiswa yang gagal menyelesaikan pendidikan.

### Permasalahan Bisnis

1. **Deteksi Dini**: Belum ada sistem yang mampu mengidentifikasi mahasiswa berisiko dropout sebelum terlambat.
2. **Faktor Pendorong**: Manajemen belum mengetahui faktor-faktor utama yang mendorong mahasiswa untuk dropout.
3. **Monitoring**: Tidak ada dashboard terpusat untuk memantau performa dan tren dropout mahasiswa secara real-time.

### Cakupan Proyek

- Eksplorasi dan analisis mendalam data mahasiswa (EDA)
- Preprocessing data dan feature engineering
- Pembangunan dan evaluasi model machine learning untuk prediksi dropout
- Pembuatan business dashboard di Looker Studio
- Deployment model sebagai prototype Streamlit yang dapat diakses publik

### Persiapan

**Sumber data:**  
Dataset Students' Performance dari Jaya Jaya Institut:  
https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance

**Setup environment:**

```bash
# 1. Clone repository
git clone <your-repo-url>
cd jaya-jaya-institut

# 2. Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Jalankan notebook
jupyter notebook notebook.ipynb

# 5. Jalankan Streamlit app
streamlit run app.py
```

---

## Business Dashboard

Dashboard dibuat menggunakan **Looker Studio** untuk memudahkan manajemen Jaya Jaya Institut dalam:
- Memantau distribusi dan tren status mahasiswa (Graduate, Enrolled, Dropout)
- Menganalisis faktor-faktor utama yang berkorelasi dengan dropout
- Memonitor performa akademik per semester
- Melacak distribusi demografis mahasiswa (usia, gender, beasiswa, dll.)

📊 **Link Dashboard:**  
[Jaya Jaya Institut — Student Performance Dashboard](https://lookerstudio.google.com/your-dashboard-link)

> **Cara Akses:**  
> Dashboard dapat diakses secara publik melalui link di atas tanpa login.

**Elemen visualisasi yang tersedia:**
- Pie chart distribusi status mahasiswa
- Bar chart dropout rate per program studi
- Scatter plot nilai masuk vs performa semester
- Tren dropout berdasarkan status keuangan
- Heatmap korelasi fitur akademik

---

## Menjalankan Sistem Machine Learning

### Cara Menjalankan Prototype Lokal

```bash
# Pastikan sudah di direktori proyek dan semua dependensi terinstall
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

### Fitur Prototype

1. **Prediksi Individual** — Masukkan data satu mahasiswa, sistem memberikan:
   - Probabilitas dropout (%)
   - Tingkat risiko (Tinggi / Sedang / Rendah)
   - Rekomendasi tindakan spesifik

2. **Prediksi Batch** — Upload file CSV berisi banyak mahasiswa, sistem akan:
   - Memprediksi seluruh data sekaligus
   - Menampilkan ringkasan distribusi risiko
   - Menyediakan hasil untuk diunduh kembali

### Akses Online

🚀 **Link Streamlit Cloud:**  
[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

> Deploy ke Streamlit Community Cloud:
> 1. Push repository ke GitHub
> 2. Buka [share.streamlit.io](https://share.streamlit.io)
> 3. Hubungkan repo dan set `app.py` sebagai main file
> 4. Klik **Deploy**

---

## Conclusion

Berdasarkan hasil analisis dan pemodelan machine learning pada data Jaya Jaya Institut:

1. **Tingkat Dropout** mencapai sekitar **28–32%** — ini adalah masalah signifikan yang membutuhkan intervensi sistematis.

2. **Faktor Akademik** adalah prediktor terkuat. Mahasiswa yang tidak lulus ≥50% mata kuliah di semester 1 memiliki probabilitas dropout jauh lebih tinggi.

3. **Faktor Finansial** sangat berperan. Mahasiswa dengan tunggakan biaya kuliah atau berstatus debitur memiliki dropout rate lebih tinggi dibanding yang pembayarannya lancar.

4. **Penerima Beasiswa** terbukti memiliki dropout rate lebih rendah, menunjukkan bantuan finansial efektif mencegah dropout.

5. **Model Terbaik** (Logistic Regression / Random Forest / Gradient Boosting tergantung hasil run) dipilih berdasarkan ROC-AUC dan Recall untuk kelas Dropout, karena dalam konteks ini **false negative (gagal deteksi mahasiswa yang akan dropout) lebih berbahaya** dibanding false positive.

---

## Rekomendasi Action Items

- **Action Item 1 — Early Warning Otomatis:** Integrasikan model prediksi ini ke sistem informasi akademik. Setiap awal semester, mahasiswa dengan probabilitas dropout >60% otomatis masuk daftar prioritas bimbingan konselor.

- **Action Item 2 — Program Pendampingan Akademik Intensif:** Mahasiswa yang tidak lulus >50% mata kuliah di semester 1 langsung dijadwalkan sesi konsultasi wajib dengan dosen pembimbing akademik dalam 2 minggu pertama semester 2.

- **Action Item 3 — Solusi Finansial Proaktif:** Bentuk tim "Financial Aid Fast Track" yang menghubungi mahasiswa dengan tunggakan sebelum akhir semester, menawarkan opsi cicilan, potongan biaya, atau bantuan beasiswa darurat.

- **Action Item 4 — Program Mentoring Peer:** Pasangkan mahasiswa berisiko tinggi dengan mahasiswa senior berprestasi sebagai mentor. Data menunjukkan dukungan sosial dan akademik dari sesama mahasiswa efektif meningkatkan retensi.

- **Action Item 5 — Evaluasi Program Studi Bermasalah:** Identifikasi program studi dengan dropout rate tertinggi dan lakukan evaluasi kurikulum, beban studi, dan kualitas pengajaran secara berkala.