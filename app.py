import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
import base64

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

st.title("🩺 AI Diagnostic PCG Analyzer (Gemini Only)")
st.markdown("Upload a heart sound file (.wav or .mp3) and get diagnostic interpretation using Gemini AI.")

uploaded_file = st.file_uploader("Upload PCG Audio File", type=["wav", "mp3"])
if uploaded_file:
    file_bytes = uploaded_file.read()
    
    # Convert audio to base64 for Gemini input
    b64_audio = base64.b64encode(file_bytes).decode()

    st.audio(file_bytes, format="audio/wav")

    # Gemini prompt (audio is just described here)
    prompt = f"""
    This is a phonocardiogram (PCG) heart sound recording. Based on listening and medical context, analyze this heart sound and provide:
    - Likely diagnosis (Normal, Murmur, Extra Sound, etc.)
    - Differential diagnosis
    - Suggested clinical steps
    Assume this is a real patient case and use clinical reasoning.
    """

    if st.button("🧠 Get Gemini AI Diagnosis"):
        response = model.generate_content(prompt)
        st.markdown("### 📋 Gemini AI Report")
        st.write(response.text)
