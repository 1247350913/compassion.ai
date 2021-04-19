                                ### Streamlit App ###

# Imports
import streamlit as st
from pages import home
from pages import models
from pages import survey
from pages import results


# Layout Style
st.set_page_config(page_title="Care Hub", layout="wide", page_icon=":heart:", initial_sidebar_state='auto')


# Sidebar

# Image and Header
from PIL import Image
image = Image.open('images/heart.jpg')
st.sidebar.image(image, caption="We Are Care Hub :)")

# Navigation Pane
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(label="Go To:", options=['Home', 'Models', 'Survey', 'Results'])

# Contact Pane
st.sidebar.title("Contact Us")
st.sidebar.text("email: carehub@gmail.com")
st.sidebar.text("github: https://github.com/camjhirsh/mental_health")


# Run All Pages
if navigation == 'Home':
    home.home_page()
elif navigation == 'Models':
    models.run_models()
elif navigation == 'Survey':
    survey.generate_survey()
elif navigation == 'Results':
    results.display_results()
