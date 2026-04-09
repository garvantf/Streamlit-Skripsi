import streamlit as st
import joblib
import pandas as pd
import os

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Uji Coba Model Sentimen",
    layout="wide"
)

st.title("Uji Coba Model Analisis Sentimen")
st.caption("Model: SVM + TF-IDF (hasil training Google Colab)")
st.markdown("---")

# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "models/model_manual_smote_C1.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model tidak ditemukan. Pastikan file ada di folder models/")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Gagal load model: {e}")
    st.stop()

st.success("Model berhasil dimuat")

# ===============================
# PILIH MODE PREDIKSI
# ===============================
mode = st.radio(
    "Pilih Mode Prediksi:",
    ["Prediksi Teks Tunggal", "Prediksi File CSV"]
)

# ===============================
# MODE 1 — TEKS TUNGGAL
# ===============================
if mode == "Prediksi Teks Tunggal":

    text = st.text_area(
        "Masukkan teks ulasan:",
        placeholder="Contoh: Aplikasi sangat membantu dan dokter cepat merespon"
    )

    if st.button("🔍 Prediksi"):
        if text.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            pred = model.predict([text])[0]

            if pred == 0:
                st.success("Sentimen: **POSITIF**")
            elif pred == 2:
                st.error("Sentimen: **NEGATIF**")
            else:
                st.info(f"Hasil prediksi: {pred}")

# ===============================
# MODE 2 — FILE CSV
# ===============================
else:
    uploaded_file = st.file_uploader(
        "Upload file CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview Data")
        st.dataframe(df.head())

        text_column = st.selectbox(
            "Pilih kolom teks:",
            df.columns
        )

        if st.button("📊 Prediksi Seluruh Data"):
            texts = df[text_column].astype(str).tolist()
            preds = model.predict(texts)

            label_map = {
                0: "Positif",
                2: "Negatif"
            }

            df["hasil_prediksi"] = [label_map.get(p, p) for p in preds]

            st.success("✅ Prediksi selesai")
            st.dataframe(df)

            csv_result = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                csv_result,
                "hasil_prediksi_sentimen.csv",
                "text/csv"
            )
