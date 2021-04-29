                                ### Survey Page ###

# Imports
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pages import survey


def display_results():
    st.title("Results")
    st.header("Based on the responses saved, these are your results:")

    generate = st.button("Click Here to generate results")
    st.text("it may take a few moments")

    if generate:
        df = survey.get_responses()
        answers, pred = generate_model()


        st.header("Does your company take mental health seriously enough:")
        if pred:
            st.subheader("Yes They Do!! :)")
        else:
            st.subheader("Nope they don't :(")

        percent = np.sum(answers[1:]) / 15 * 100
        st.header("Your Score Is:")
        st.subheader(str(percent)[:2] + '%')

        st.header("Here are features that led to this score:")
        pie = plt.figure(figsize=(8, 6))
        plt.pie([percent, 100 - percent], labels=['% optimal responses', 'not optimal'])
        st.pyplot(pie)

        nons = [ans for ans in answers if not ans]
        st.subheader("These are the non optimals:")
        questions = np.random.choice(15, 2)
        st.write("Indicies of questions:")
        for ind in questions:
            st.write(ind)


def generate_model():

    # Step 1: Load and Clean Data

    # 2014 Survey
    mental_2014 = pd.read_csv('data/mental_2014.csv')
    mental_2014 = mental_2014.drop(labels=['Timestamp', 'Country', 'state', 'self_employed',
                                            'obs_consequence', 'comments', 'phys_health_consequence',
                                            'phys_health_interview', 'work_interfere', 'mental_health_interview'],
                                            axis=1)
    # 2016 Survey
    mental_2016 = pd.read_csv('data/mental_2016.csv')

    # 2019 survey
    mental_2019 = pd.read_csv('data/mental_2019.csv')
    mental_2019['treatment'] = mental_2019['treatment'].astype(int)
    mental_2019['tech_company'].fillna(value=False, inplace=True)
    mental_2019['tech_company'] = mental_2019['tech_company'].astype(int)
    mental_2019['mental_vs_physical'] = mental_2019['Overall, how much importance does your employer place on physical health?'] < mental_2019['Overall, how much importance does your employer place on mental health?']
    mental_2019.drop(columns=['Overall, how much importance does your employer place on physical health?', 'Overall, how much importance does your employer place on mental health?'], inplace=True)
    mental_2019['mental_vs_physical'] = mental_2019['mental_vs_physical'].replace({True:'Yes', False:'No'})

    # Combine Data
    combined = combined = pd.concat([mental_2014, mental_2016, mental_2019], axis=0)
    responses = ['Yes', 'No']
    combined = combined[combined['mental_vs_physical'].str.contains('|'.join(responses)) == True]
    combined['care_options'] = combined['care_options'].str.replace('I am not sure', 'Not sure')
    combined['care_options'].fillna(value='Not sure', inplace=True)

    # Removing 'Maybe' responses
    responses = ['Yes', 'No']
    combined = combined[combined['mental_vs_physical'].str.contains('|'.join(responses)) == True]

    # Fixing null values in 'care_options'
    combined['care_options'] = combined['care_options'].str.replace('I am not sure', 'Not sure')
    combined['care_options'].fillna(value='Not sure', inplace=True)

    combined.drop(columns=['Gender', 'remote_work'], inplace=True)
    combined.rename(columns={'Age':'age'}, inplace=True)


    # Step 2: Prep Data For Models

    from sklearn.model_selection import train_test_split

    # split
    X = pd.get_dummies(combined.drop(['mental_vs_physical'], axis=1))
    y = [1 if i == 'Yes' else 0 for i in combined['mental_vs_physical']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=88)

    # test and train
    train = X_train.copy()
    train['mental_vs_physical'] = y_train
    test = X_test.copy()
    test['mental-vs_physical'] = y_test

    # calculate TPR and FPR
    def TPR(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm.ravel()[3]/ (cm.ravel()[3] + cm.ravel()[2])
    def FPR(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm.ravel()[1]/ (cm.ravel()[1] + cm.ravel()[0])

    grid_values = {'max_features': np.linspace(1,8,8, dtype='int32'),
                   'min_samples_leaf': [5],
                   'n_estimators': [500],
                   'random_state': [88]}

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import confusion_matrix

    # grid search for max_features
    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, param_grid=grid_values, cv=5, verbose=1)
    rf_cv.fit(X_train, y_train)


    y_pred_rf = rf_cv.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    ind = np.random.choice(len(y_pred_rf))
    return X_test.iloc[ind,:], y_pred_rf[ind]
