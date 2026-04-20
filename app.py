import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jaya Jaya Institut — Dropout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f0f4f8; }
    
    /* Global Font Color */
    html, body, [class*="css"], p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: blue !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #1a237e 0%, #283593 60%, #3949ab 100%);
    }
    [data-testid="stSidebar"] * { color: #e8eaf6 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label { color: #c5cae9 !important; }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h2 { font-size: 2rem; margin: 4px 0; font-weight: 700; }
    .metric-card p { color: #64748b; font-size: 0.85rem; margin: 0; }

    .risk-high   { background: linear-gradient(135deg,#fee2e2,#fecaca); border-left: 5px solid #ef4444; }
    .risk-medium { background: linear-gradient(135deg,#fef9c3,#fde68a); border-left: 5px solid #f59e0b; }
    .risk-low    { background: linear-gradient(135deg,#dcfce7,#bbf7d0); border-left: 5px solid #22c55e; }

    .result-box {
        border-radius: 16px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
    }
    .result-box h1 { font-size: 3rem; margin: 0; }
    .result-box h3 { font-size: 1.1rem; margin: 6px 0 0; font-weight: 500; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e3a5f;
        margin: 20px 0 8px;
        padding-bottom: 4px;
        border-bottom: 2px solid #3b82f6;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }

    /* Hide streamlit default elements */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model & artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("model/best_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/feature_cols.json") as f:
        features = json.load(f)
    with open("model/model_info.json") as f:
        info = json.load(f)
    return model, scaler, features, info

try:
    model, scaler, feature_cols, model_info = load_artifacts()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"⚠️ Model tidak ditemukan: {e}\nPastikan folder `model/` tersedia.")
    st.stop()


# ── Helper ────────────────────────────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.65:
        return "TINGGI", "risk-high", "🔴"
    elif prob >= 0.40:
        return "SEDANG", "risk-medium", "🟡"
    else:
        return "RENDAH", "risk-low", "🟢"

COURSE_MAP = {
    33:"Biofuel Production Technologies", 171:"Animation & Multimedia Design",
    8014:"Social Service (evening)", 9003:"Agronomy", 9070:"Communication Design",
    9085:"Veterinary Nursing", 9119:"Informatics Engineering",
    9130:"Equiniculture", 9147:"Management", 9238:"Social Service",
    9254:"Tourism", 9500:"Nursing", 9556:"Oral Hygiene",
    9670:"Advertising & Marketing Management", 9773:"Journalism & Communication",
    9853:"Basic Education", 9991:"Management (evening)"
}

MARITAL_MAP = {1:"Single",2:"Married",3:"Widower",4:"Divorced",5:"Facto Union",6:"Legally Separated"}
QUAL_MAP    = {1:"Secondary Edu",2:"Higher Edu Bachelors",3:"Higher Edu Degree",4:"Higher Edu Masters",
               5:"Higher Edu Doctorate",6:"Frequency Higher Edu",9:"12th Year Not Completed",
               10:"11th Year Not Completed",12:"Other - 11th Year",14:"10th Year",
               15:"10th Year Not Completed",19:"Basic Edu 3rd Cycle",38:"Basic Edu 2nd Cycle",
               39:"Technological Specialization",40:"Higher Edu Degree (1st cycle)",
               42:"Professional Higher Technical",43:"Higher Edu Masters (2nd cycle)"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("🎓 Jaya Jaya Institut")
    st.caption("Dropout Early Warning System")
    st.divider()

    st.markdown("### 📊 Model Info")
    st.info(
        f"**Model:** {model_info['best_model_name']}\n\n"
        f"**ROC-AUC:** {model_info['roc_auc']:.3f}\n\n"
        f"**F1 Dropout:** {model_info['f1_dropout']:.3f}\n\n"
        f"**Recall Dropout:** {model_info['recall_dropout']:.3f}"
    )
    st.divider()
    st.markdown(
        "<small>💡 Sistem ini membantu mendeteksi mahasiswa berisiko dropout "
        "sehingga bisa segera diberikan bimbingan.</small>",
        unsafe_allow_html=True
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a5f,#2563eb);
            padding:28px 32px; border-radius:16px; margin-bottom:24px; color:white;'>
    <h1 style='margin:0;font-size:2rem;'>🎓 Student Dropout Prediction</h1>
    <p style='margin:6px 0 0; opacity:0.85; font-size:1rem;'>
        Jaya Jaya Institut — Early Warning System untuk Deteksi Mahasiswa Berisiko
    </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Individual", "📋 Prediksi Batch (CSV)", "ℹ️ Panduan"])

# ============================================================
# TAB 1 — Individual Prediction
# ============================================================
with tab1:
    st.markdown("### Masukkan Data Mahasiswa")
    st.caption("Isi seluruh data di bawah, lalu klik **Prediksi Sekarang**.")

    # ── SECTION 1: Informasi Pribadi ─────────────────────────
    st.markdown('<p class="section-header">👤 Informasi Pribadi</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x==0 else "Laki-laki")
        age    = st.slider("Usia Saat Enrollment", 17, 70, 20)
        marital = st.selectbox("Status Perkawinan", list(MARITAL_MAP.keys()),
                               format_func=lambda x: MARITAL_MAP[x])
    with c2:
        nationality = st.selectbox("Kewarganegaraan (kode)", [1,2,6,11,13,14,17,21,22,24,25,26,32,41,62,100,101,103,105,108,109], index=0)
        international = st.selectbox("Mahasiswa Internasional", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        displaced = st.selectbox("Displaced (Pindahan)", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
    with c3:
        scholarship = st.selectbox("Penerima Beasiswa", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        debtor = st.selectbox("Status Debitur (Tunggakan)", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        tuition_up  = st.selectbox("Biaya Kuliah Up-to-Date", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")
        edu_special = st.selectbox("Kebutuhan Khusus Pendidikan", [0,1], format_func=lambda x: "Tidak" if x==0 else "Ya")

    # ── SECTION 2: Akademik ──────────────────────────────────
    st.markdown('<p class="section-header">📚 Latar Belakang Akademik</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        course = st.selectbox("Program Studi", list(COURSE_MAP.keys()),
                              format_func=lambda x: COURSE_MAP[x])
        prev_qual = st.selectbox("Kualifikasi Sebelumnya", list(QUAL_MAP.keys()),
                                 format_func=lambda x: QUAL_MAP[x])
        prev_qual_grade = st.number_input("Nilai Kualifikasi Sebelumnya", 0.0, 200.0, 130.0, 0.5)
    with c2:
        admission_grade = st.number_input("Nilai Masuk (Admission Grade)", 0.0, 200.0, 130.0, 0.5)
        app_mode  = st.selectbox("Mode Aplikasi", [1,2,5,7,10,15,16,17,18,26,27,39,42,43,44,51,53,57], index=2)
        app_order = st.selectbox("Urutan Pilihan (1=pilihan pertama)", list(range(1,10)))
    with c3:
        mothers_q   = st.selectbox("Kualifikasi Ibu", list(QUAL_MAP.keys()), format_func=lambda x: QUAL_MAP[x], index=0)
        fathers_q   = st.selectbox("Kualifikasi Ayah", list(QUAL_MAP.keys()), format_func=lambda x: QUAL_MAP[x], index=0)
        mothers_occ = st.selectbox("Pekerjaan Ibu (kode 0-10)", list(range(11)), index=5)
        fathers_occ = st.selectbox("Pekerjaan Ayah (kode 0-10)", list(range(11)), index=5)
        daytime = st.selectbox("Kelas", [1,0], format_func=lambda x: "Pagi/Siang" if x==1 else "Malam")

    # ── SECTION 3: Performa Semester ─────────────────────────
    st.markdown('<p class="section-header">📝 Performa Semester</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.caption("**Semester 1**")
        cu1_cred  = st.number_input("MK Dikreditkan Sem 1", 0, 20, 0)
        cu1_enr   = st.slider("MK Diambil Sem 1", 0, 10, 5)
        cu1_eval  = st.slider("MK Dievaluasi Sem 1", 0, 20, 6)
        cu1_app   = st.slider("MK Lulus Sem 1", 0, 10, 4)
        cu1_grade = st.number_input("Nilai Rata-rata Sem 1 (0–20)", 0.0, 20.0, 12.0, 0.1)
        cu1_noev  = st.number_input("MK Tanpa Evaluasi Sem 1", 0, 10, 0)
    with c2:
        st.caption("**Semester 2**")
        cu2_cred  = st.number_input("MK Dikreditkan Sem 2", 0, 20, 0)
        cu2_enr   = st.slider("MK Diambil Sem 2", 0, 10, 5)
        cu2_eval  = st.slider("MK Dievaluasi Sem 2", 0, 20, 6)
        cu2_app   = st.slider("MK Lulus Sem 2", 0, 10, 4)
        cu2_grade = st.number_input("Nilai Rata-rata Sem 2 (0–20)", 0.0, 20.0, 12.0, 0.1)
        cu2_noev  = st.number_input("MK Tanpa Evaluasi Sem 2", 0, 10, 0)

    # ── SECTION 4: Kondisi Ekonomi ───────────────────────────
    st.markdown('<p class="section-header">💹 Kondisi Ekonomi Makro</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: unemp     = st.number_input("Tingkat Pengangguran (%)", 0.0, 30.0, 11.0, 0.1)
    with c2: inflation = st.number_input("Tingkat Inflasi (%)", -5.0, 15.0, 1.0, 0.1)
    with c3: gdp       = st.number_input("GDP", -5.0, 10.0, 1.0, 0.1)

    st.divider()
    predict_btn = st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame([{
            'Marital_status': marital, 'Application_mode': app_mode,
            'Application_order': app_order, 'Course': course,
            'Daytime_evening_attendance': daytime, 'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade, 'Nacionality': nationality,
            'Mothers_qualification': mothers_q, 'Fathers_qualification': fathers_q,
            'Mothers_occupation': mothers_occ, 'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade, 'Displaced': displaced,
            'Educational_special_needs': edu_special, 'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_up, 'Gender': gender,
            'Scholarship_holder': scholarship, 'Age_at_enrollment': age,
            'International': international,
            'Curricular_units_1st_sem_credited': cu1_cred,
            'Curricular_units_1st_sem_enrolled': cu1_enr,
            'Curricular_units_1st_sem_evaluations': cu1_eval,
            'Curricular_units_1st_sem_approved': cu1_app,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_1st_sem_without_evaluations': cu1_noev,
            'Curricular_units_2nd_sem_credited': cu2_cred,
            'Curricular_units_2nd_sem_enrolled': cu2_enr,
            'Curricular_units_2nd_sem_evaluations': cu2_eval,
            'Curricular_units_2nd_sem_approved': cu2_app,
            'Curricular_units_2nd_sem_grade': cu2_grade,
            'Curricular_units_2nd_sem_without_evaluations': cu2_noev,
            'Unemployment_rate': unemp, 'Inflation_rate': inflation, 'GDP': gdp
        }])

        X_scaled = scaler.transform(input_data[feature_cols])
        prob = model.predict_proba(X_scaled)[0][0] # Dropout is class 0
        pred = model.predict(X_scaled)[0]
        level, css_class, icon = get_risk_level(prob)

        # Result box
        if pred == 0:
            box_style = "background:linear-gradient(135deg,#fee2e2,#fca5a5);border:2px solid #ef4444;"
            verdict = "⚠️ BERISIKO DROPOUT"
            color   = "#991b1b"
        else:
            box_style = "background:linear-gradient(135deg,#dcfce7,#86efac);border:2px solid #22c55e;"
            verdict = "✅ TIDAK BERISIKO DROPOUT"
            color   = "#14532d"

        st.markdown(f"""
        <div class="result-box" style="{box_style}">
            <h1 style="color:{color};">{icon} {level}</h1>
            <h3 style="color:{color};">{verdict}</h3>
            <p style="font-size:1.5rem; font-weight:700; color:{color}; margin-top:8px;">
                Probabilitas Dropout: {prob*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Metric breakdown
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            pass_rate_1 = cu1_app / max(cu1_enr, 1) * 100
            st.metric("Tingkat Lulus Sem 1", f"{pass_rate_1:.0f}%",
                      delta="OK" if pass_rate_1 >= 60 else "Perlu Perhatian",
                      delta_color="normal" if pass_rate_1 >= 60 else "inverse")
        with col_b:
            pass_rate_2 = cu2_app / max(cu2_enr, 1) * 100
            st.metric("Tingkat Lulus Sem 2", f"{pass_rate_2:.0f}%",
                      delta="OK" if pass_rate_2 >= 60 else "Perlu Perhatian",
                      delta_color="normal" if pass_rate_2 >= 60 else "inverse")
        with col_c:
            fin_ok = tuition_up and not debtor
            st.metric("Status Keuangan", "Aman ✓" if fin_ok else "Perlu Cek ⚠️")

        # Recommendations
        if pred == 1:
            st.warning("### 📋 Rekomendasi Tindakan")
            recs = []
            if cu1_app / max(cu1_enr,1) < 0.5 or cu2_app / max(cu2_enr,1) < 0.5:
                recs.append("📚 **Akademik**: Jadwalkan konseling akademik segera — tingkat kelulusan mata kuliah di bawah 50%.")
            if not tuition_up or debtor:
                recs.append("💰 **Finansial**: Hubungi bagian keuangan untuk program cicilan atau bantuan beasiswa darurat.")
            if age > 30:
                recs.append("🧑‍🎓 **Dukungan**: Mahasiswa dewasa sering butuh fleksibilitas jadwal — pertimbangkan kelas malam atau online.")
            if not scholarship:
                recs.append("🏆 **Beasiswa**: Cek kelayakan mahasiswa untuk program beasiswa prestasi atau kebutuhan.")
            if not recs:
                recs.append("🤝 **Mentoring**: Pasangkan dengan mentor senior untuk dukungan motivasional dan akademik.")
            for r in recs:
                st.markdown(f"- {r}")


# ============================================================
# TAB 2 — Batch Prediction
# ============================================================
with tab2:
    st.markdown("### Prediksi Batch dari File CSV")
    st.info(
        "Upload file CSV dengan kolom yang sama seperti dataset (separator `;`). "
        "Sistem akan memprediksi risiko dropout untuk setiap mahasiswa."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded, sep=';')
            st.success(f"✅ File berhasil diupload: {df_upload.shape[0]} mahasiswa, {df_upload.shape[1]} kolom.")
            st.dataframe(df_upload.head(), use_container_width=True)

            missing_cols = [c for c in feature_cols if c not in df_upload.columns]
            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
            else:
                X_batch = scaler.transform(df_upload[feature_cols])
                probs = model.predict_proba(X_batch)[:,0]
                preds = model.predict(X_batch)

                df_result = df_upload.copy()
                df_result['Dropout_Probability'] = (probs * 100).round(1)
                df_result['Predicted_Status'] = ['Dropout' if p==0 else 'Non-Dropout' for p in preds]
                df_result['Risk_Level'] = [get_risk_level(p)[0] for p in probs]

                st.divider()
                st.markdown("### Hasil Prediksi")

                # Summary metrics
                c1, c2, c3, c4 = st.columns(4)
                n = len(df_result)
                n_high   = sum(probs >= 0.65)
                n_medium = sum((probs >= 0.40) & (probs < 0.65))
                n_low    = sum(probs < 0.40)
                c1.metric("Total Mahasiswa", n)
                c2.metric("🔴 Risiko Tinggi",  n_high,  f"{n_high/n*100:.1f}%")
                c3.metric("🟡 Risiko Sedang", n_medium, f"{n_medium/n*100:.1f}%")
                c4.metric("🟢 Risiko Rendah",   n_low,  f"{n_low/n*100:.1f}%")

                st.dataframe(
                    df_result[['Dropout_Probability','Predicted_Status','Risk_Level'] + feature_cols[:5]],
                    use_container_width=True
                )

                csv_out = df_result.to_csv(index=False, sep=';').encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil Prediksi",
                    data=csv_out, file_name="hasil_prediksi_dropout.csv",
                    mime="text/csv", use_container_width=True
                )

        except Exception as e:
            st.error(f"Error memproses file: {e}")


# ============================================================
# TAB 3 — Panduan
# ============================================================
with tab3:
    st.markdown("### ℹ️ Panduan Penggunaan Sistem")

    with st.expander("📌 Tentang Sistem Ini", expanded=True):
        st.markdown("""
        Sistem **Early Warning Dropout** ini dibangun untuk membantu Jaya Jaya Institut
        mendeteksi mahasiswa yang berisiko mengalami dropout sedini mungkin.

        **Cara Kerja:**
        1. Data mahasiswa dimasukkan melalui form atau upload CSV
        2. Model machine learning memproses data dan menghitung probabilitas dropout
        3. Hasil ditampilkan lengkap dengan tingkat risiko dan rekomendasi tindakan

        **Tingkat Risiko:**
        - 🔴 **TINGGI** (≥65%): Perlu intervensi segera
        - 🟡 **SEDANG** (40–64%): Perlu pemantauan aktif
        - 🟢 **RENDAH** (<40%): Pantau secara berkala
        """)

    with st.expander("📊 Faktor-Faktor Penting"):
        st.markdown("""
        Berdasarkan analisis model, faktor-faktor paling berpengaruh terhadap dropout:

        | Faktor | Dampak |
        |--------|--------|
        | Jumlah MK yang lulus di Sem 1 & 2 | Sangat Tinggi |
        | Nilai rata-rata semester | Sangat Tinggi |
        | Status biaya kuliah (up-to-date) | Tinggi |
        | Status debitur (tunggakan) | Tinggi |
        | Usia saat enrollment | Sedang |
        | Penerima beasiswa | Sedang |
        | Tingkat pengangguran nasional | Rendah |
        """)

    with st.expander("🚀 Menjalankan Lokal"):
        st.code("""
# 1. Clone / download project
# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
        """, language="bash")

    st.divider()
    st.caption("© 2024 Jaya Jaya Institut | Dibuat untuk keperluan submission Data Science Dicoding")
