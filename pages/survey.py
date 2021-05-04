                                ### Survey Page ###

# Imports
import streamlit as st
import pandas as pd

# df = pd.DataFrame(columns=['age', 'family_history', 'treatment', 'no_employees', 'tech_company', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence', 'coworkers', 'supervisor', 'mental_vs_physical'])

row = []

def generate_survey():
    st.title("Employee Satisfaction Survey")
    st.header("How's everything going for you?")
    st.subheader("Let us know how you feel about how your company manages its employees' mental health.")
    st.subheader("We'll use your feedback to make this a better place to work.")
    st.text("15 questions: Takes 4 min")

    q1 = st.text_input("What is your age?")
    q2 = st.radio("Do you have any family history of mental health issues?", ['No', 'Yes', "I don't know"])
    q3 = st.radio("Have you ever seeked treatment for mental health concerns?", ["No", "Yes"])
    q4 = st.radio("Number of employees in company", ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
    q5 = st.radio("Are you in a tech company?", ["Yes", "No"])
    q6 = st.radio("Does your company offer you benefits?", ["Yes", "No", "idk", "Not eligible"])
    q7 = st.radio("Does your company offer and cover mental health care options", ["Yes", "No", "Not sure"])
    q8 = st.radio("Does your company offer a wellness program", ["Yes", "No", "Not sure"])
    q9 = st.radio("Have you ever seeked help for mental health concerns?", ["Yes", "No", "Not sure"])
    q10 = st.radio("Does your company offer anonymity for mental health consultation?", ["Yes", "No", "Not sure"])
    q11 = st.radio("How easily does your company offer leave of absence?", ['Difficult', 'Not sure', 'Very difficult', 'Very easy', 'Neither', 'Somewhat Difficult', "Somewhat Easy"])
    q12 = st.radio("Have you ever had mental health consequences?", ["Yes", "No", "Not sure"])
    q13 = st.radio("Do you feel you can confide in your coworkers?", ["Yes", "No", "Maybe", "Some of them"])
    q14 = st.radio("Do you feel you can confide in your supervisor?", ["Maybe", "No", "Some of them", "Yes"])
    q15 = st.radio("Do you think your company cares about your mental health as much as your physical presence", ["Yes", "No"])


    save_ans = st.button("Click Here to save responses")

    if save_ans:
        row.append(q1)
        if q2 == 'Yes':
            row.append(0)
            row.append(0)
            row.append(1)
        elif q2 == 'No':
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(1)
            row.append(0)
            row.append(0)
        if q3 == 'Yes':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        else:
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        if q4 == '1-5':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q4 == '100-500':
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q4 == '26-100':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q4 == '500-1000':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        elif q4 =="6-25":
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q5 == 'No':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q6 == 'Not eligible':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        elif q6 == "idk":
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q6 == 'Yes':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        else:
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        if q7 == "No":
            row.append(1)
            row.append(0)
            row.append(0)
        elif q7 == "Not sure":
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(1)
        if q8=='Not sure':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q8=='No':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q9=='Not sure':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q9=='No':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q10=='Not sure':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q10=='No':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q11=='Difficult':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q11=='Not sure':
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q11=='Neither':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q11=='Somewhat Difficult':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q11=='Somewhat Easy':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        elif q11=='Very Difficult':
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q12=='Not sure':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q12=='No':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q13=='Maybe':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q13=='No':
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        elif q13=='Some of them':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)
        if q14=='Maybe':
            row.append(1)
            row.append(0)
            row.append(0)
            row.append(0)
        elif q14=='No':
            row.append(0)
            row.append(1)
            row.append(0)
            row.append(0)
        elif q14=='Some of them':
            row.append(0)
            row.append(0)
            row.append(1)
            row.append(0)
        else:
            row.append(0)
            row.append(0)
            row.append(0)
            row.append(1)


def get_responses():
    return row
