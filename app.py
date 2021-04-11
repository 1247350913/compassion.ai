# Streamlit App #

# Import Necessary Libraries
import streamlit as st
import pandas as pd
import numpy as np

# Title and Headings
st.title("Product Title")

st.header("""Does your company take the mental health of its employees as seriously as physical health?""")

st.subheader("Mental Health in the Tech Industry")

st.text(""" These are some stats on the state of mental health in tech rn""")

st.subheader("What [our product] can do your your employees")

st.text(""" This is what we can do for your company and people""")

# Load data
df = pd.read_csv("survey.csv")

# Display survey
st.write(df)


st.line_chart(df)
