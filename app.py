import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# CONFIGURASI HALAMAN UTAMA WEB
st.set_page_config(
    page_title="Gen Z Impulse Buying Analytics Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pembuatan style CSS antarmuka
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; text-align: center; color: #4B5563; margin-bottom: 25px; }
    .section-holder { padding: 15px; border-radius: 8px; background-color: #F3F4F6; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# OPTIMIZED ENGINE: LOAD & PROCESS DATA
@st.cache_data
def load_and_process_dataset():
    # Membaca berkas utama dataset kuesioner
    df = pd.read_excel("Raw data_Impulse buying behavior.xlsx", sheet_name='Dataset')
    
    # Kalkulasi nilai rata-rata fungsional per dimensi variabel
    df['SC_Avg'] = df[['SC1', 'SC2', 'SC3', 'SC4']].mean(axis=1)
    df['SI_Avg'] = df[['SI1', 'SI2', 'SI3', 'SI4', 'SI5']].mean(axis=1)
    df['TR_Avg'] = df[['TR1', 'TR2', 'TR3', 'TR4', 'TR5']].mean(axis=1)
    df['HM_Avg'] = df[['HM1', 'HM2', 'HM3']].mean(axis=1)
    df['SL_Avg'] = df[['SL1', 'SL2', 'SL3', 'SL4']].mean(axis=1)
    df['PP_Avg'] = df[['PP1', 'PP2', 'PP3', 'PP4']].mean(axis=1)
    df['OIB_Avg'] = df[['OIB1', 'OIB2', 'OIB3']].mean(axis=1)
    
    # Penentuan threshold target biner berbasis Median Split
    median_threshold = df['OIB_Avg'].median()
    df['Target_Impulsive'] = np.where(df['OIB_Avg'] >= median_threshold, 1, 0)
    
    # Feature Selection & Penyelarasan Skala
    feature_cols = ['SC_Avg', 'SI_Avg', 'TR_Avg', 'HM_Avg', 'SL_Avg', 'PP_Avg']
    X = df[feature_cols]
    y = df['Target_Impulsive']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Inisialisasi dan pelatihan model terbaik
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)
    
    return scaler, model, df, feature_cols

scaler, model, df_clean, features = load_and_process_dataset()

# BANNER STRUKTUR HEADER UTAMA
st.markdown("<div class='main-title'>🛍️ Dashboard Analisis Prediktif Perilaku Belanja Impulsif Gen Z</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Sistem Pendukung Keputusan Bisnis Digital — Tugas Besar Penambangan Data Telkom University Surabaya</div>", unsafe_allow_html=True)

# SIDEBAR: PANEL INFORMASI PROYEK KELOMPOK
st.sidebar.header("📂 Informasi Proyek")
st.sidebar.markdown("""
**Metodologi:** CRISP-DM  
**Ukuran Sampel:** 361 Responden  
**Target Populasi:** Gen Z (Regional ASEAN)  
**Algoritma Utama:** Logistic Regression  
**Akurasi Pengujian:** **71.23%** """)
st.sidebar.divider()
st.sidebar.caption("Dikembangkan oleh Kelompok Tugas Besar Data Mining © 2026")

# MEMBENTUK STRUKTUR MENU TAB INTERAKTIF
tab1, tab2, tab3 = st.tabs([
    "📊 Eksplorasi & Distribusi Data", 
    "📈 Evaluasi Eksperimen Model", 
    "🔮 Modul Simulasi Prediksi Baru"
])

# TAB 1: EKSPLORASI DATA & RINGKASAN DATASET
with tab1:
    st.subheader("📋 Analisis Deskriptif Data Primer Kuesioner")
    st.write("Bagian ini menampilkan ringkasan struktural dari 361 data responden yang berhasil dikumpulkan kelompok.")
    
    # Tampilan ringkasan metrik statistika deskriptif utama
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Responden Valid", f"{len(df_clean)} Data", "100%")
    m2.metric("Kategori Impulsif Tinggi (Kelas 1)", f"{df_clean['Target_Impulsive'].sum()} Responden", "58.17%")
    m3.metric("Kategori Rasional (Kelas 0)", f"{len(df_clean) - df_clean['Target_Impulsive'].sum()} Responden", "41.83%")
    
    st.divider()
    
    col_graph, col_preview = st.columns([4, 5])
    
    with col_graph:
        st.markdown("**Visualisasi Sebaran Kelas Target**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Target_Impulsive', data=df_clean, palette=['#10B981', '#EF4444'], ax=ax)
        ax.set_xticklabels(['0: Rasional', '1: Impulsif Tinggi'])
        ax.set_xlabel("Kategori Perilaku")
        ax.set_ylabel("Jumlah Responden")
        st.pyplot(fig)
        plt.close(fig)
        
    with col_preview:
        st.markdown("**Preview Sampel Data Rekayasa Fitur (Rata-rata Likert)**")
        display_cols = features + ['OIB_Avg', 'Target_Impulsive']
        st.dataframe(df_clean[display_cols].head(10), height=230, use_container_width=True)
        
        # PERBAIKAN & PENAMBAHAN LEGEND EXPLANATION KOLOM
        with st.expander("🔍 Klik di sini untuk membaca penjelasan arti nama kolom tabel"):
            st.markdown("""
            * **`SC_Avg` (*Social Comparison*):** Nilai rata-rata kecenderungan konsumen membandingkan gaya hidup dengan orang lain di media sosial.
            * **`SI_Avg` (*Social Influence*):** Nilai rata-rata kuatnya pengaruh teman, keluarga, maupun *influencer* terhadap keputusan belanja.
            * **`TR_Avg` (*Trust*):** Nilai rata-rata tingkat rasa aman dan kepercayaan konsumen terhadap sistem pembayaran dan keaslian toko digital.
            * **`HM_Avg` (*Hedonic Motivation*):** Nilai rata-rata dorongan belanja yang didasari oleh pencarian kesenangan atau menghilangkan stres.
            * **`SL_Avg` (*Store Layout*):** Nilai rata-rata kenyamanan visual, desain estetika, dan kemudahan navigasi antarmuka aplikasi belanja.
            * **`PP_Avg` (*Price Promotion*):** Nilai rata-rata tingkat sensitivitas konsumen terhadap pemicu diskon, *flash sale*, atau gratis ongkir.
            * **`OIB_Avg` (*Online Impulse Buying*):** Nilai rata-rata murni perilaku belanja spontan sebelum dikelompokkan ke kelas target.
            * **`Target_Impulsive`:** Hasil pembagian kelas berdasarkan *Median Split* (**0** = Rasional jika nilai OIB < 4.0; **1** = Impulsif Tinggi jika nilai OIB ≥ 4.0).
            """)

# TAB 2: EVALUASI EKSPERIMEN MODEL
with tab2:
    st.subheader("📈 Hasil Evaluasi Komparatif Arsitektur Kecerdasan Buatan")
    st.write("Pembuktian empiris performa model klasifikasi prediktif melalui perbandingan matriks pengujian.")
    
    col_el, col_cm = st.columns(2)
    
    with col_el:
        st.markdown("#### **1. Optimasi Klastering Melalui Elbow Method**")
        try:
            st.image("elbow_plot.png", use_container_width=True)
            st.info("💡 **Interpretasi Siku:** Grafik WCSS melandai secara konsisten tepat setelah titik klaster k=3, mengonfirmasi segmentasi perilaku alami terbagi optimal ke dalam 3 kelompok.")
        except:
            st.warning("Berkas 'elbow_plot.png' tidak ditemukan di direktori lokal. Pastikan file main.py telah dieksekusi.")
            
    with col_cm:
        st.markdown("#### **2. Perbandingan Akurasi & Confusion Matrix (Data Uji 20%)**")
        try:
            st.image("confusion_matrix_comparison.png", use_container_width=True)
            st.success("💡 **Analisis Model:** Algoritma Logistic Regression unggul dengan akurasi 71.23% dibandingkan Naïve Bayes Classifier yang mencatat akurasi 69.86%.")
        except:
            st.warning("Berkas 'confusion_matrix_comparison.png' tidak ditemukan di direktori lokal.")

# TAB 3: MODUL SIMULASI PREDIKSI BARU
with tab3:
    st.subheader("🔮 Pengujian Karakteristik Konsumen Secara Real-Time")
    st.write("Geser panel parameter di bawah ini untuk mensimulasikan nilai evaluasi psikografis konsumen baru.")
    
    c_input, c_result = st.columns([5, 4])
    
    with c_input:
        st.markdown("<div class='section-holder'><b>🎛️ Panel Kontrol Stimulus Konsumen (Skala Likert 1.0 - 5.0)</b></div>", unsafe_allow_html=True)
        sc = st.slider("Social Comparison (Kecenderungan membandingkan diri di media sosial)", 1.0, 5.0, 3.0, 0.1)
        si = st.slider("Social Influence (Daya pengaruh kelompok referensi/KOL/influencer)", 1.0, 5.0, 3.0, 0.1)
        tr = st.slider("Trust (Tingkat kepercayaan terhadap sistem dan keamanan platform)", 1.0, 5.0, 3.0, 0.1)
        hm = st.slider("Hedonic Motivation (Motivasi belanja demi hiburan/mereduksi stres)", 1.0, 5.0, 3.0, 0.1)
        sl = st.slider("Store Layout (Penilaian estetika visual navigasi antarmuka e-commerce)", 1.0, 5.0, 3.0, 0.1)
        pp = st.slider("Price Promotion (Tingkat sensitivitas terhadap diskon dan flash sale)", 1.0, 5.0, 3.0, 0.1)
        
    with c_result:
        st.markdown("<div class='section-holder'><b>📊 Hasil Analisis Prediktif Kecerdasan Sistem</b></div>", unsafe_allow_html=True)
        
        # Penyelarasan format input baru ke dimensi array NumPy
        raw_input = np.array([[sc, si, tr, hm, sl, pp]])
        scaled_input = scaler.transform(raw_input)
        
        # Eksekusi fungsi prediksi dan ekstraksi nilai probabilitas model
        pred_class = model.predict(scaled_input)[0]
        pred_probabilities = model.predict_proba(scaled_input)[0]
        
        st.write("---")
        if pred_class == 1:
            st.error("🚨 **PREDIKSI KLASIFIKASI: HIGH IMPULSE BUYER (IMPULSIF TINGGI)**")
            st.metric(label="Probabilitas Perilaku Spontan", value=f"{pred_probabilities[1]*100:.2f}%")
            st.markdown("""
            **🎯 Rekomendasi Intervensi Pemasaran (Agresif):**
            * Aktifkan trigger psikologis kelangkaan seperti pop-up *countdown flash sale* berbatas waktu singkat.
            * Sediakan voucher pemotong harga instan atau penawaran *bundle* komplementer sebelum pengguna keluar dari halaman keranjang belanja.
            """)
        else:
            st.success("✅ **PREDIKSI KLASIFIKASI: RATIONAL BUYER (IMPULSIF RENDAH)**")
            st.metric(label="Probabilitas Pertimbangan Logis", value=f"{pred_probabilities[0]*100:.2f}%")
            st.markdown("""
            **🎯 Rekomendasi Intervensi Pemasaran (Kognitif):**
            * Tampilkan visualisasi ulasan/testimoni bintang 5 yang kredibel serta kartu jaminan spesifikasi produk secara detail.
            * Tonjolkan fitur garansi pengembalian dana penuh atau jaminan keaslian barang untuk mengeliminasi keraguan belanja konsumen.
            """)
        st.write("---")