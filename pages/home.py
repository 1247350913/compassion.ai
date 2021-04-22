                                ### Home Page ###

# Imports
import streamlit as st


def home_page():

    c1, c2, c3 = st.beta_columns(3)
    c2.title("compassion.ai")
    st.header("""Does your company take the mental health of its employees as seriously as physical health?""")

    st.subheader("Mental Health in the Tech Industry")
    st.text(""" These are some stats on the state of mental health in tech rn""")
    st.subheader("What [our product] can do your your employees")
    st.text(""" This is what we can do for your company and people""")
