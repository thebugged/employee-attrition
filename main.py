import streamlit as st
from streamlit_option_menu import option_menu

from apps.home import home_page
from apps.predict import predict_page
from apps.insights import insights_page

st.set_page_config(
    page_title="RetentionIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# navigation menu
selected = option_menu(
    menu_title = None,
    options=["Home", "Predict", "Insights"],
    icons=["house", "graph-up-arrow", "robot"],
    orientation="horizontal",
    default_index=0
    )

# routing
if selected == "Home":
    home_page()
elif selected == "Predict":
    predict_page()
elif selected == "Insights":
    insights_page()

