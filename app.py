import os
import subprocess
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def load_components():
    model = joblib.load("C:/Users/AtikahDR/Documents/Data Science Project/CyberBullying/model_svm.pkl")
    vectorizer = joblib.load("C:/Users/AtikahDR/Documents/Data Science Project/CyberBullying/tfidf_vectorizer.pkl")
    label_encoder = joblib.load("C:/Users/AtikahDR/Documents/Data Science Project/CyberBullying/label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_components()


st.set_page_config(
    page_title="Cyberbullying Detection App",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Cyberbullying Detection App")
st.markdown("""
            Aplikasi ini menggunakan model Machine Learning untuk mendeteksi **Jenis Cyberbullying** dari teks.
            Masukkan teks di bawah ini dan tekan tombol **Prediksi** untuk melihat hasil klasifikasi """)

user_input = st.text_area("📝 Masukkan teks:", height=150, placeholder="Ketik teks atau tweet di sini...")

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        # Transform teks ke bentuk vektor TF - IDF
        X_input = vectorizer.transform([user_input])

        # Prediksi Kelas
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)

        # Ambil daftar label yang dikenali oleh model
        allowed_labels = model.classes_
        
        # Konversi hasil prediksi ke label asli
        label_pred = le.inverse_transform(y_pred)[0] if hasattr(le, "inverse_transform") else y_pred[0]

        # Mengambil probabilitas tiap kelas
        probs = dict(zip(le.classes_, y_prob[0] * 100))

        # Tampilan Hasil
        st.subheader("🎯 Hasil Prediksi")
        st.success(f"**Jenis Cyebrbullying:** {label_pred}")

        # Tampilan probabilitas tiap kelas (Bar Chart)
        st.markdown("### 📊 Probabilitas Kelas")
        st.bar_chart(probs)

        # Info Tambahan
        st.caption("Model: Support Vector Machine (SVM) - Tuned with TF-IDF features")

# Footer
st.markdown("---")
st.markdown("💡 Atikah Dr")