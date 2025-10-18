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
    model = joblib.load("model_logreg.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, le = load_components()

# Konfigurasi Page
st.set_page_config(
    page_title="Cyberbullying Detection App",
    page_icon="💬",
    layout="wide",
)

# Sidebar Navigation
st.sidebar.title("💬 Cyberbullying Detection")
page = st.sidebar.radio("Navigasi", ["🧠 Prediction", "📊 Bar Probabilitas", "🧾 History"])

st.sidebar.markdown("---")
st.sidebar.caption("Created by **Atikah DR**")

# Inisialisasi session_state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "last_label" not in st.session_state:
    st.session_state.last_label = None

# Page 1: Prediction
if page == "🧠 Prediction":
    st.title("💬 Cyberbullying Detection App")
    st.markdown("""
        This application uses a Machine Learning model to detect types of cyberbullying from text.
        Enter the text below and press the Predict button to see the classification results. 
    """)

    user_input = st.text_area("📝 Enter text:", height=150, placeholder="Type your text or tweet here...")

if st.button("🔍 Prediction"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter the text first.")
    else:
        # Transform text into vector form TF - IDF
        X_input = vectorizer.transform([user_input])

        # Prediction of Class
        y_pred = model.predict(X_input)
        y_prob = model.predict_proba(X_input)

        # Converting prediction results to original labels
        label_pred = le.inverse_transform(y_pred)[0] if hasattr(le, "inverse_transform") else y_pred[0]

        # Taking the probability of each class
        probs = dict(zip(le.classes_, y_prob[0] * 100))

        # Save results to session
        st.session_state.last_probs = probs
        st.session_state.last_label = label_pred

         # Save to History
        st.session_state.history.append({
            "Teks": user_input,
            "Result": label_pred
        })
        
        # Results Display
        st.subheader("🎯 Prediction Results")
        if label_pred.lower() == "not_cyberbullying":
            st.success(f"✅ **Types of Cyberbullying:** {label_pred}")
        else:
            st.error(f"🚨 **Types of Cyberbullying:** {label_pred}")
            
        st.markdown("---")
        st.caption("Model: Logistic Regression - Tuned with TF-IDF features")

# Page 2: Probabilitas (Bar Chart)
elif page == "📊 Bar Probabilitas":
    st.title("📊 Probability Class Prediction")

    if st.session_state.last_probs is not None:
        st.markdown(f"### The final text is classified as: **{st.session_state.last_label}**")
        st.bar_chart(st.session_state.last_probs)
    else:
        st.info("⚠️ No prediction results yet. Please make a prediction first in the menu. **Prediction**.")

# Page 3: History
elif page == "🧾 History":
    st.title("🧾Previous Detection History")

    if len(st.session_state.history) > 0:
        history_df = pd.DataFrame(st.session_state.history)
        history_df["Result"] = history_df["Result"].apply(
            lambda x: f"✅ {x}" if x.lower() == "not_cyberbullying" else f"🚨 {x}"
        )

        st.dataframe(history_df[::-1], use_container_width=True)  

    # Tombol Delete History
    if st.button("🗑️ Delete History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("📭 No prediction history yet.")


#  Footer
st.markdown("---")
st.caption("💡 Created by Atikah DR | Machine Learning Cyberbullying Project")




