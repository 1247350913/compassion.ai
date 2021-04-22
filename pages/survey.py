                                ### Survey Page ###

# Imports
import streamlit as st


def generate_survey():
    st.title("Employee Satisfaction Survey")
    st.header("How's everything going for you?")
    st.subheader("Let us know how you feel about how your company manages its employees' mental health.")
    st.subheader("We'll use your feedback to make this a better place to work.")
    st.text("Takes 4 min")
    generate = st.button("OK, let's get started")
    display_questions = False

    if generate:
        display_questions = True

    if display_questions:

        st.write('make the survey and save answers to df that can be plugged into model')

        st.text_input("What year were you born?")

        st.radio("What is your gender identity?", ["No Answer", "Male", "Female", "Non-binary", "Other"])

        st.radio("Are you self-employed?", ["No Answer", "Yes", "No"])
