import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

st.title("🩺 AI PCG Interpretation (Text-Based Only)")
st.markdown("Write PCG observations manually and get diagnosis.")

prompt = st.text_area("Enter PCG findings (e.g., murmurs, timing, location):")

if st.button("Get Diagnosis"):
    response = model.generate_content(prompt)
    st.markdown("### 📋 Gemini AI Report")
    st.write(response.text)
