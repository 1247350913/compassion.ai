                                ### Streamlit App ###

# Imports
import streamlit as st
from pages import home
from pages import survey
from pages import results
from pages import about


# Layout Style
st.set_page_config(page_title="compassion.ai", page_icon=":heart:", initial_sidebar_state='auto')


# Sidebar

# Image and Header
from PIL import Image
image = Image.open('images/logo_2.png').resize(size=(100, 100))

c1, c2, c3 = st.beta_columns(3)
with c2:
    st.sidebar.image(image) # caption="We Are compassion.ai :)")

# Navigation Pane
st.sidebar.title("Navigation")
navigation = st.sidebar.radio(label="Go To:", options=['Home', 'Survey', 'Results'])

# Contact Pane
st.sidebar.title("Contact Us")
sign = st.sidebar.button("Sign In")
about_us = st.sidebar.button("About Us")

# Run All Pages
if about_us:
    about.display_about()

if navigation == 'Home' and not about_us:
    home.home_page()
elif navigation == 'Survey' and not about_us:
    survey.generate_survey()
elif navigation == 'Results' and not about_us:
    results.display_results()
