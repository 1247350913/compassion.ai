                                ### Streamlit App ###

# Imports
import streamlit as st
from pages import home
from pages import models
from pages import survey
from pages import results


# Layout Style
st.set_page_config(page_title="Care Hub", page_icon=":heart:", initial_sidebar_state='auto')


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
st.sidebar.text("Sign In")
st.sidebar.text("About Us")


# Run All Pages
if navigation == 'Home':
    home.home_page()
# elif navigation == 'Models':
#     models.run_models()
elif navigation == 'Survey':
    survey.generate_survey()
elif navigation == 'Results':
    results.display_results()
