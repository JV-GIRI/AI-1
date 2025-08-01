import streamlit as st
import os
import numpy as np
from keras.models import load_model
from utils.audio_processing import extract_features
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

cnn_model = load_model("model/pcg_cnn_model.h5")

st.title("🎧 AI Diagnostic PCG Analyzer")
st.markdown("Upload a heart sound file (.wav or .mp3) to analyze.")

uploaded_file = st.file_uploader("Upload PCG Audio", type=["wav", "mp3"])
if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.audio("temp.wav")

    features = extract_features("temp.wav")
    prediction = cnn_model.predict(features)
    class_index = np.argmax(prediction)
    classes = ["Normal", "Murmur", "Extra Sounds", "Abnormal"]
    diagnosis = classes[class_index]

    st.success(f"🩺 **Prediction**: {diagnosis}")

    prompt = f"""
    This heart sound is classified as: {diagnosis}.
    Provide a possible diagnosis, differential causes, and clinical next steps.
    """

    if st.button("🧠 Get AI Interpretation"):
        response = model.generate_content(prompt)
        st.markdown("### 📋 Gemini AI Diagnostic Report")
        st.write(response.text)
