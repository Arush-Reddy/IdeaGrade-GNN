"""Backward-compatible Streamlit entry point.

Deployments should target ``streamlit_app.py``. Existing local commands using
``streamlit run app.py`` continue to work.
"""

from streamlit_app import *  # noqa: F401,F403
