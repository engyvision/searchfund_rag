# config.py
import os
from dotenv import load_dotenv

# Try to load Streamlit secrets first (works on Streamlit Cloud)
try:
    import streamlit as st
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise KeyError("OPENAI_API_KEY not found in Streamlit secrets.")
except (ModuleNotFoundError, KeyError, FileNotFoundError):
    # If not on Streamlit or secrets not found, fall back to .env (works locally)
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in .env or Streamlit secrets.")
