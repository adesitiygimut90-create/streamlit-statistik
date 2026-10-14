"""Web statistik (tema burgundy) yang dibungkus jadi aplikasi Streamlit.
Jalankan lokal:  streamlit run streamlit_web.py
"""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Statistik Deskriptif & Regresi", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")
# hilangkan jarak bawaan Streamlit supaya web tampil penuh
st.markdown("<style>.block-container{padding:0!important;max-width:100%!important}"
            "header[data-testid='stHeader']{display:none}</style>", unsafe_allow_html=True)

html = (Path(__file__).parent / "web_burgundy.html").read_text(encoding="utf-8")
st.iframe(html, height=1200)
