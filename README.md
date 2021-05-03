# compassion.ai
Compassion.ai is a brand new machine learning tool that will help companies assess their current mental health climate and provide recommendations for the best ways to be more accomodating for their workers.

## mental_health/data
This folder contains the csv files used to build our models. All data are gathered from Open Sourcing Mental Illness, a non-profit, dedicated to raising awareness, educating, and providing resources to support mental wellness in the tech and open source communities. The data used come from three different csv files, for the years 2014, 2016, and 2019.

## mental_health/MentalHealth-FinalCode.ipynb
This notebook contains the code used to create the model to predict whether or not a company takes mental health as seriously as physical health. The notebook starts with cleaning the csv files to be able to join together the three years of survey data. After cleaning the data, we built 7 different models: a baseline model, logistic regression, a decision tree, vanilla bagging, random forest, linear SVM, and RBF SVM. We then evaluated their performances and found that the random forest model had the highest accuracy and TPR, and the lowest FPR out of all our fitted models. This is the final model that we decided to send to the frontend for our web application. Lastly, we looked at the most important features for the model as well as general correlation for the training set.

## mental_health/images
Contains our logos for the website

All code for web application hosted by streamlit in app.py
csv files gathered from [this website].

## mental_health/app.py
This is the Streamlit application written in python with the Streamlit module. This file is hosted using Streamlit Sharing servies.

## mental_health/pages
This directory contains the files written in python using Streamlit API to display each of the pages for the application. The survey page displays the survey and records the responses into a database. The home and About fiels display information about our product. The Results page loads our ML solution, and inputs the survey responses from the database to acheive a prediction and to analyze results.

## braches
The master branch is the minimum viable product according to UI display and the new_feature branch is implementing upgrades to the UI.
