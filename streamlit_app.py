"""Streamlit entry point:  streamlit run streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # works without `pip install`

from ostutor.ui.app import main  # noqa: E402

main()
