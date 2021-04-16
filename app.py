                                ### Streamlit App ###

# Imports
import streamlit as st
import home
import models
import survey


# Layout Style
st.set_page_config(page_title="Care Hub", layout="wide", page_icon=":heart:", initial_sidebar_state='auto')


# Navigation Pane
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(label="Go To:", options=['Main', 'Models', 'Survey'])

# Contact Pane
st.sidebar.title("Contact Us")
st.sidebar.text("email: carehub@gmail.com")




# Run All Pages
if navigation == 'Main':
    home.home_page()
elif navigation == 'Models':
    models.run_models()
elif navigation == 'Survey':
    survey.generate_survey()
