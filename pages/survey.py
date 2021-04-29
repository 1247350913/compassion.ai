                                ### Survey Page ###

# Imports
import streamlit as st
import pandas as pd

df = pd.DataFrame(columns=['index', 'age', 'family_history', 'treatment', 'no_employees', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_vs_physical'])

def generate_survey():
    st.title("Employee Satisfaction Survey")
    st.header("How's everything going for you?")
    st.subheader("Let us know how you feel about how your company manages its employees' mental health.")
    st.subheader("We'll use your feedback to make this a better place to work.")
    st.text("15 questions: Takes 4 min")

    q1 = st.text_input("What is your age?")
    q2 = st.radio("Do you have any family history of mental health issues?", ['No', 'Yes', "I don't know"])
    q3 = st.radio("Have you ever seeked treatment for mental health concerns?", ["No", "Yes"])
    q4 = st.radio("Number of employees in company", ["No Answer", "1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    q5 = st.radio("Are you in a tech company?", ["Yes", "No"])
    q6 = st.radio("Does your company offer you benefits?", ["Yes", "No"])
    q7 = st.radio("Does your company offer and cover mental health care options", ["Yes", "No", "Not sure"])
    q8 = st.radio("Does your company offer a wellness program", ["Yes", "No", "Not sure"])
    q9 = st.radio("Have you ever seeked help for mental health concerns?", ["Yes", "No", "Not sure"])
    q10 = st.radio("Does your company offer anonymity for mental health consultation?", ["Yes", "No", "Not sure"])
    q11 = st.radio("How easily does your company offer leave of absence?", ['Somewhat easy', 'Somewhat difficult', "Don't know", 'Very difficult', 'Very easy', 'Neither easy nor difficult', "I don't know", 'Difficult'])
    q12 = st.radio("Have you ever had mental health consequences?", ["Yes", "No", "Maybe"])
    q13 = st.radio("Do you feel you can confide in your coworkers?", ["Yes", "No", "Maybe"])
    q14 = st.radio("Do you feel you can confide in your supervisor?", ["Yes", "No"])
    q15 = st.radio("Does your company care about your mental health as much as your physical presence", ["Yes", "No"])


    save_ans = st.button("Click Here to save responses")

    if save_ans:
        df.loc[len(df.index)] = [0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]


def get_responses():
    return df
