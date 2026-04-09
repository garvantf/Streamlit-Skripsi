import streamlit as st

st.set_page_config(page_title="Skripsi ML – Dashboard", layout="wide")

st.title("📚 Dashboard Penelitian Skripsi")
st.markdown("---")

st.header("🔍 Gambaran Umum Penelitian")
st.write("""
Aplikasi ini menampilkan hasil penelitian tugas akhir mengenai **Analisis Sentimen pada Ulasan Pengguna 
Aplikasi Telemedicine Halodoc** menggunakan:

- **Labeling Manual**
- **Labeling Leksikon**
- **SMOTE** dan **Non-SMOTE**
- **SVM** dengan variasi nilai **C = 0.01, 0.1, 1, 10**

Total terdapat **16 model** hasil pengujian yang dapat dilihat performanya dan dicoba secara langsung.
""")

st.subheader("📦 Struktur Pengujian")
st.write("""
- **Skenario A** → Manual + Non-SMOTE  
- **Skenario B** → Leksikon + Non-SMOTE  
- **Skenario C** → Manual + SMOTE  
- **Skenario D** → Leksikon + SMOTE  
""")

st.subheader("📈 Halaman Tersedia")
st.write("""
1. **Hasil Skenario** → Menampilkan hasil akurasi, precision, recall, f1, confusion matrix dari 16 model  
2. **Uji Coba Model** → User dapat memilih model dan memberikan input teks untuk prediksi  
3. **Penutup / Kesimpulan Penelitian**
""")

st.success("Sistem siap digunakan. Silakan pilih halaman dari sidebar.")
