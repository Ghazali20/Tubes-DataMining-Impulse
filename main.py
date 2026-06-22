import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# STEP 1: LOAD & PRE-PROCESSING DATA
print("--- [Step 1] Loading dan Pre-processing Data ---")
# Membaca file excel dataset kuesioner
file_name = "Raw data_Impulse buying behavior.xlsx"
df = pd.read_excel(file_name, sheet_name='Dataset')

# Rekayasa Fitur: Menghitung rata-rata skor per variabel (Sumbu X)
df['SC_Avg'] = df[['SC1', 'SC2', 'SC3', 'SC4']].mean(axis=1)
df['SI_Avg'] = df[['SI1', 'SI2', 'SI3', 'SI4', 'SI5']].mean(axis=1)
df['TR_Avg'] = df[['TR1', 'TR2', 'TR3', 'TR4', 'TR5']].mean(axis=1)
df['HM_Avg'] = df[['HM1', 'HM2', 'HM3']].mean(axis=1)
df['SL_Avg'] = df[['SL1', 'SL2', 'SL3', 'SL4']].mean(axis=1)
df['PP_Avg'] = df[['PP1', 'PP2', 'PP3', 'PP4']].mean(axis=1)
df['OIB_Avg'] = df[['OIB1', 'OIB2', 'OIB3']].mean(axis=1)

# Transformasi Target Biner (Sumbu Y) menggunakan Median Split
median_threshold = df['OIB_Avg'].median()
df['Target_Impulsive'] = np.where(df['OIB_Avg'] >= median_threshold, 1, 0)

# Memisahkan Fitur Sumbu X dan Target Sumbu Y
X_features = df[['SC_Avg', 'SI_Avg', 'TR_Avg', 'HM_Avg', 'SL_Avg', 'PP_Avg']]
y_target = df['Target_Impulsive']

# Standardisasi Fitur Sumbu X (Wajib untuk K-Means & LogReg)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print(f"Data Berhasil Diproses! Nilai Median Threshold OIB: {median_threshold}")
print(f"Jumlah Target Impulsif (Kelas 1) : {sum(y_target == 1)} responden")
print(f"Jumlah Target Rasional (Kelas 0)  : {sum(y_target == 0)} responden\n")

# STEP 2: CLUSTERING (K-MEANS & ELBOW METHOD)
print("--- [Step 2] Menjalankan K-Means & Elbow Method ---")
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Grafik Elbow dan menyimpannya sebagai file gambar
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method untuk Menentukan K Optimal (Gen Z Impulse Buying)')
plt.xlabel('Jumlah Cluster (k)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.savefig('elbow_plot.png') # Otomatis tersimpan di folder proyek
print("Grafik 'elbow_plot.png' berhasil disimpan.\n")

# Mengelompokkan responden ke dalam 3 klaster optimal
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# STEP 3: CLASSIFICATION (SUPERVISED LEARNING)
print("--- [Step 3] Melatih Model Klasifikasi Prediktif ---")
# Split data menjadi 80% Training dan 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_target, test_size=0.2, random_state=42)

# A. Algoritma Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# B. Algoritma Naïve Bayes Classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

print(f"Akurasi Model Logistic Regression : {acc_lr * 100:.2f}%")
print(f"Akurasi Model Naïve Bayes Classifier: {acc_nb * 100:.2f}%\n")

# STEP 4: EVALUATION (CONFUSION MATRIX PLOT)
print("--- [Step 4] Membuat Visualisasi Perbandingan Evaluasi ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Matriks Evaluasi Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Confusion Matrix - Logistic Regression\n(Akurasi: {acc_lr*100:.1f}%)')
axes[0].set_xlabel('Label Prediksi')
axes[0].set_ylabel('Label Aktual')

# Matriks Evaluasi Naïve Bayes
cm_nb = confusion_matrix(y_test, y_pred_nb)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
axes[1].set_title(f'Confusion Matrix - Naïve Bayes\n(Akurasi: {acc_nb*100:.1f}%)')
axes[1].set_xlabel('Label Prediksi')
axes[1].set_ylabel('Label Aktual')

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png') # Otomatis tersimpan di folder proyek
print("Grafik 'confusion_matrix_comparison.png' berhasil disimpan.\n")
print("--- Selesai! Semua visualisasi dan perhitungan siap untuk Bab IV ---")