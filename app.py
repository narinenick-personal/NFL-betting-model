"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nfl_model.dashboard import render_dashboard

render_dashboard(ROOT)
