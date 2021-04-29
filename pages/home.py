                                ### Home Page ###

# Imports
import streamlit as st
from PIL import Image


def home_page():

    c1, c2, c3 = st.beta_columns(3)
    with c1:
        st.title("compassion.ai")
    with c3:
        image = Image.open('images/logo_2.png').resize(size=(100, 100))
        st.image(image)

    st.header("""Helping employers help employees""")


    st.subheader("Mental Health in the Tech Industry")
    st.write("""These are some stats on the state of mental health in tech rn""")

    st.subheader("How It Works")
    st.write("""Compassion.ai is a tool for companies to better understand their employees values when it\ncomes to mental health. Through machine learning techniques and real data from employees,\nwe are able to predict the major obstacles for employees present in the tech workspace,\nthus empowering companies to make their environments better.\nHelping employers help employees.""")
    st.markdown("Step 1: Fill Out Survey")
    st.markdown("Step 2: Run our Model in Results")
    st.markdown("Step 3: Analyze Results and Improve Work Environment")
