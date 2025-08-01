import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import librosa
import numpy as np

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

st.title("💓 AI Diagnostic PCG Analyzer (Text-Based, Gemini-Pro)")
st.markdown("Upload a heart sound (.wav or .mp3), auto extract info, and get diagnosis.")

uploaded_file = st.file_uploader("📂 Upload PCG Audio File", type=["wav", "mp3"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    st.audio(file_bytes, format="audio/wav")

    # Save file to temporary location
    with open("temp_audio.wav", "wb") as f:
        f.write(file_bytes)

    # Load audio
    y, sr = librosa.load("temp_audio.wav")
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = np.mean(librosa.feature.rms(y=y))

    # Create a descriptive text
    audio_description = f"""
    - Duration: {duration:.2f} seconds
    - Sampling Rate: {sr} Hz
    - Average RMS (Intensity): {rms:.6f}
    - Estimated Heartbeat Tempo: {tempo:.2f} BPM
    """

    st.markdown("### 📊 Audio Description")
    st.code(audio_description)

    prompt = f"""
    This is a description of a phonocardiogram (PCG) recording:

    {audio_description}

    Based on this and typical heart sound analysis, provide:
    - Likely diagnosis (Normal, Murmur, Extra Sound, etc.)
    - Differential diagnosis
    - Suggested clinical actions

    Respond like a cardiologist.
    """

    if st.button("🧠 Diagnose with Gemini"):
        response = model.generate_content(prompt)
        st.markdown("### 📋 Gemini AI Diagnosis")
        st.write(response.text)
