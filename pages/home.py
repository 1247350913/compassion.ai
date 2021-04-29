                                ### Home Page ###

# Imports
import streamlit as st


def home_page():

    # c1, c2, c3 = st.beta_columns(3)
    # with c2:
    st.title("compassion.ai")
    st.header("""Helping employers help employees""")

    st.subheader("Mental Health in the Tech Industry")
    st.text(""" These are some stats on the state of mental health in tech rn""")
    st.subheader("How It Works")
    st.text("""Compassion.ai is a tool for companies to better understand their employees values when it comes\nto mental health. Through machine learning techniques and real data from employees, we are able to predict the major obstacles for employees present in the tech workspace, thus empowering companies to make their environments better. Helping employers help employees.""")
