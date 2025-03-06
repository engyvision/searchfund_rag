# config.py
import os
from dotenv import load_dotenv

# Try to load Streamlit secrets first (works on Streamlit Cloud)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
# If not on Streamlit, fall back to .env (works locally)
except ModuleNotFoundError:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
