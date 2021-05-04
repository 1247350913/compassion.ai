                                ### Survey Page ###

# Imports
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pages import survey


def display_results():
    st.title("Results")
    st.header("Based on the responses saved, these are your results:")

    generate = st.button("Click Here to generate results")
    st.text("it may take a few moments")

    if generate:
        row = survey.get_responses()
        xt, model = generate_model()
        xt.loc[len(xt.index)] = row
        df = xt.iloc[len(xt.index)-1:]
        prediction = model.predict(df)

        st.header("Does your company take mental health seriously enough:\n")

        if not prediction:
            st.subheader("\n Our model predicted that your company does not :(")
        else:
            st.subheader("\n Our model predicted that your company does :)")

        optimal_response = [0]*58
        best = [1,4,7,12,16,22,28,36,42,48,52,57]
        for i in range(58):
            if i in best:
                optimal_response[i] = 1

        count = 0
        for ind in range(len(optimal_response)):
            if row[ind] == 1 and optimal_response[i] == 1:
                count += 1

        pie = plt.figure(figsize=(8, 6))
        plt.pie([count, 100 - count], labels=['% optimal responses', 'not optimal'])
        st.pyplot(pie)


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

    return X_test, rf_cv
