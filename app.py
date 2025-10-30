import os
import subprocess
import xgboost
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def load_components():
    model = joblib.load("model_xgb.pkl")
    vectorizer = joblib.load("tfidf_vectorizerx.pkl")
    label_encoder = joblib.load("label_encoderx.pkl")
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
page = st.sidebar.radio("Navigasi", ["🧠 Prediction", "📊 Bar Probabilitas", "🕸️ Radar Chart", "🧾 History"])

st.sidebar.markdown("---")
st.sidebar.caption("Created by **Atikah DR**")

# Inisialisasi session_state
if "history" not in st.session_state:
    st.session_state.history = []
if "last_probs" not in st.session_state:
    st.session_state.last_probs = None
if "last_label" not in st.session_state:
    st.session_state.last_label = None
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
            probs = dict(zip(le.inverse_transform(model.classes_), y_prob[0] * 100))

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
    st.header("📊 Probability Class Prediction")

    if st.session_state.last_probs is not None and len(st.session_state.last_probs) > 0:
        st.markdown(f"### The final text is classified as: **{st.session_state.last_label}**")

        probs_df = pd.DataFrame(
            list(st.session_state.last_probs.items()),
            columns=["Class", "Probability"]
        ).sort_values("Probability", ascending=False)

        st.bar_chart(probs_df.set_index("Class"))
    else:
        st.warning("⚠️ No prediction results yet. Please make a prediction first in the **Prediction** menu.")


# Page 3: Radar Chart
elif page == "🕸️ Radar Chart":
    st.header("🕸️ Radar Visualization of Cyberbullying Prediction")

    if st.session_state.last_probs is not None and len(st.session_state.last_probs) > 0:
        probs_df = pd.DataFrame(list(st.session_state.last_probs.items()), columns=["Class", "Probability"])
        probs_df = probs_df.sort_values("Class")

        categories = probs_df["Class"].tolist()
        values = probs_df["Probability"].tolist()
        values += [values[0]]  # Menutup lingkaran
        categories += [categories[0]]

        fig = go.Figure()

        # Garis utama radar
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            mode='lines+markers',
            line=dict(color='royalblue', width=3),
            marker=dict(size=8, color='azure', symbol='circle'),
            fill='toself',
            fillcolor='rgba(30,144,255,0.15)',  # Biru lembut transparan
            name='Cyberbullying Probability'
        ))

        # Gaya layout spider chart
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',  # background transparan
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='lightgray',      # garis jaring spider
                    gridwidth=1,
                    showline=True,
                    linewidth=1,
                    linecolor='gray',
                    tickfont=dict(size=14, color='white'),
                    ticks='outside'
                ),
                angularaxis=dict(
                    gridcolor='lightgray',      # garis jaring spider arah sudut
                    gridwidth=1,
                    tickfont=dict(size=14, color='white')
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',  # seluruh background transparan
            font=dict(size=16, color='white'),
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ No prediction results yet. Please make a prediction first in the **Prediction** menu.")

# Page 4: History
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
        st.toast("History deleted successfully!", icon="🗑️")

else:
    st.info("📭 No prediction history yet.")


#  Footer
st.markdown("---")

st.caption("💡 Created by Atikah DR | Machine Learning Prediction Project")
