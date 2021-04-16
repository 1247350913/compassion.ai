                                ### Models to Display ###


# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def run_models():

    c1, c2, c3 = st.beta_columns(3)
    with c2:
        st.title('Model Info')
    st.subheader("This is an in-depth look at how we built our product using [the surveys]")
    c1, c2, c3 = st.beta_columns(3)
    with c2:
        load = st.button("Click Here to Load Models")
    st.text("This will take a while to run, please do not leave this section until it finishes")
    if load:
        outputs = get_model_outputs()
        st.pyplot(outputs[0])
        st.pyplot(outputs[1])
        st.write(outputs[2])
        st.pyplot(outputs[3])
        st.pyplot(outputs[4])




def get_model_outputs():

    # Output Array Types: [plt, plt, df, plt, plt]
    outputs = []

    # Load and Clean Data
    # 2014 Survey
    mental_2014 = pd.read_csv('mental_2014.csv')
    mental_2014 = mental_2014.drop(labels=['Timestamp', 'Country', 'state', 'self_employed',
                                            'obs_consequence', 'comments', 'phys_health_consequence',
                                            'phys_health_interview', 'work_interfere', 'mental_health_interview'],
                                            axis=1)
    # 2016 Survey
    mental_2016 = pd.read_csv('mental_2016.csv')

    # Combine Data
    combined = pd.concat([mental_2014, mental_2016], axis=0)
    responses = ['Yes', 'No']
    combined = combined[combined['mental_vs_physical'].str.contains('|'.join(responses)) == True]
    combined['care_options'] = combined['care_options'].str.replace('I am not sure', 'Not sure')
    combined['care_options'].fillna(value='Not sure', inplace=True)
    combined.drop('Gender', axis=1, inplace=True)


    # Prep Data Sets for Models

    # split combined data into test and training
    from sklearn.model_selection import train_test_split

    # split
    X = pd.get_dummies(combined.drop(['mental_vs_physical'], axis=1))
    y = [1 if i == 'Yes' else 0 for i in combined['mental_vs_physical']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=88)

    # make train
    train = X_train.copy()
    train['mental_vs_physical'] = y_train

    # make test
    test = X_test.copy()
    test['mental-vs_physical'] = y_test

    # calculate TPR and FPR
    def TPR(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm.ravel()[3]/ (cm.ravel()[3] + cm.ravel()[2])
    def FPR(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm.ravel()[1]/ (cm.ravel()[1] + cm.ravel()[0])


    # Logistic Regression

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score

    # fit log reg model and get confusion matrix
    logreg = LogisticRegression(random_state=88)
    logreg.fit(X_train, y_train)

    y_prob = logreg.predict_proba(X_test)
    y_pred_logreg = pd.Series([1 if x > 0.5 else 0 for x in y_prob[:,1]])

    cm_logreg = confusion_matrix(y_test, y_pred_logreg)


    # Decision Tree

    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    # first try grid search for best ccp_alpha
    grid_values = {'ccp_alpha': np.linspace(0, 0.1, 51)}

    dtc = DecisionTreeClassifier(random_state=88)
    dtc_cv = GridSearchCV(dtc, param_grid=grid_values, cv=5).fit(X_train, y_train)

    # plot ccp_alpha values from grid search first try
    ccp_alpha = dtc_cv.cv_results_['param_ccp_alpha'].data
    ACC_scores = dtc_cv.cv_results_['mean_test_score']

    plt1 = plt.figure(figsize=(8, 6))
    plt.xlabel('ccp_alpha', fontsize=16)
    plt.ylabel('CV Accuracy', fontsize=16)
    plt.scatter(ccp_alpha, ACC_scores, s=3)
    plt.plot(ccp_alpha, ACC_scores, linewidth=3)
    plt.grid(True, which='both')

    plt.tight_layout()

    outputs.append(plt1)

    # second try grid search for best ccp_alpha
    grid_values = {'ccp_alpha': np.linspace(0, 0.01, 101)}

    dtc = DecisionTreeClassifier(random_state=88)
    dtc_cv = GridSearchCV(dtc, param_grid=grid_values, cv=5).fit(X_train, y_train)

    # plot ccp_alpha values from grid search second try
    ccp_alpha = dtc_cv.cv_results_['param_ccp_alpha'].data
    ACC_scores = dtc_cv.cv_results_['mean_test_score']

    plt2 = plt.figure(figsize=(8, 6))
    plt.xlabel('ccp_alpha', fontsize=16)
    plt.ylabel('CV Accuracy', fontsize=16)
    plt.scatter(ccp_alpha, ACC_scores, s=3)
    plt.plot(ccp_alpha, ACC_scores, linewidth=3)
    plt.grid(True, which='both')

    plt.tight_layout()

    outputs.append(plt2)

    y_pred_dtc = dtc_cv.predict(X_test)
    cm_dtc = confusion_matrix(y_test, y_pred_dtc)


    # Vanilla Bagging

    from sklearn.ensemble import RandomForestClassifier

    # random forest model with max_features set to the total number of features
    bagging = RandomForestClassifier(max_features=len(X_train.columns), random_state=1)
    bagging.fit(X_train, y_train)

    y_pred_bagging = bagging.predict(X_test)
    cm_bagging = confusion_matrix(y_test, y_pred_bagging)


    # Random Forest

    # grid search for max_features
    grid_values = {'max_features': np.linspace(1,8,8, dtype='int32'),
                   'min_samples_leaf': [5],
                   'n_estimators': [500],
                   'random_state': [88]}

    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, param_grid=grid_values, cv=5, verbose=1)
    rf_cv.fit(X_train, y_train)


    y_pred_rf = rf_cv.predict(X_test)
    cm_rf = confusion_matrix(y_test, y_pred_rf)


    # Compare Models

    comparison_data = {'Logistic Regression': [accuracy_score(y_test, y_pred_logreg),
                                               TPR(y_test, y_pred_logreg), FPR(y_test, y_pred_logreg)],
                       'Decision Tree': [accuracy_score(y_test, y_pred_dtc), TPR(y_test, y_pred_dtc),
                                         FPR(y_test, y_pred_dtc)],
                       'Vanilla Bagging': [accuracy_score(y_test, y_pred_bagging), TPR(y_test, y_pred_bagging),
                                         FPR(y_test, y_pred_bagging)],
                       'Random Forest': [accuracy_score(y_test, y_pred_rf), TPR(y_test, y_pred_rf), FPR(y_test, y_pred_rf)]}

    performance_df = pd.DataFrame(data=comparison_data, index=['Accuracy', 'TPR', 'FPR'])

    outputs.append(performance_df)

    # Importance Score for random forest

    pd.DataFrame({'Feature' : X_train.columns,
                  'Importance score': 100*rf_cv.best_estimator_.feature_importances_}).round(1).sort_values(
                        'Importance score', ascending=False)

    corr = pd.get_dummies(train).corr()[['mental_vs_physical']].sort_values('mental_vs_physical', ascending=False).iloc[1:,:]

    positive_corr = corr[corr['mental_vs_physical'] > 0]
    negative_corr = corr[corr['mental_vs_physical']<0]

    # positive correlation bar graph
    plt3 = plt.figure(figsize=(8,7))
    plt.barh(positive_corr.index, 100*positive_corr['mental_vs_physical'])
    plt.title('Positive Correlation for variables and mental_vs_physical (scaled by 100)');

    outputs.append(plt3)

    # negative correlation bar graph
    plt4 = plt.figure(figsize=(8,7))
    plt.barh(negative_corr.index, abs(100*negative_corr['mental_vs_physical']))
    plt.title('Absolute Value Negative Correlation for variables and mental_vs_physical (scaled by 100)');

    outputs.append(plt4)


    # Feature Selection

    from sklearn.feature_selection import SelectFromModel

    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
    sel.fit(X_train, y_train)

    selected_feat= X_train.columns[(sel.get_support())]

    X_train_select = X_train.drop(columns = selected_feat)
    X_test_select = X_test.drop(columns = selected_feat)


    # Random Forest with Feature Selection

    # grid search for max_features
    grid_values = {'max_features': np.linspace(1,8,8, dtype='int32'),
                   'min_samples_leaf': [5],
                   'n_estimators': [500],
                   'random_state': [88]}

    rf = RandomForestClassifier()
    rf_cv = GridSearchCV(rf, param_grid=grid_values, cv=5, verbose=1)
    rf_cv.fit(X_train_select, y_train)

    y_pred_rf = rf_cv.predict(X_test_select)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    return outputs
