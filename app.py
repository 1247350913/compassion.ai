                                ### Streamlit App ###

# Imports
import streamlit as st
import home
import models


# Layout Style
st.set_page_config(page_title="Care Hub", layout="wide", page_icon=":shark:", initial_sidebar_state='auto')


# Navigation Pane
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(label="Go To:", options=['Main', 'Models'])

if navigation == 'Main':
    home.home_page()
elif navigation == 'Models':
    models.display_title()
