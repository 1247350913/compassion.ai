# compassion.ai
Compassion.ai is a brand new machine learning application that will help companies assess their current mental health climate and provide recommendations for the best ways to be more accomodating for their workers. Within this GitHub repository are all the necessary files used to create our web application.

## mental_health/data
This folder contains the csv files used to build our models. All data are gathered from Open Sourcing Mental Illness, a non-profit, dedicated to raising awareness, educating, and providing resources to support mental wellness in the tech and open source communities. The data used come from three different csv files, for the years 2014, 2016, and 2019.

## mental_health/MentalHealth-FinalCode.ipynb
This notebook contains the code used to create the model to predict whether or not a company takes mental health as seriously as physical health. The notebook starts with cleaning the csv files to be able to join together the three years of survey data. During the cleaning process, we removed unnecessary columns, dealt with null values, and made the entries in each column consistent. After cleaning the data, we built 7 different models to predict the independent variable, mental_vs_physical which is a binary variable where a 1 means the company takes mental health as seriously as physical health and 0 others. We started with a baseline model which predicts all entries to be a negative classification. Next we used a logistic regression model, using variable selection to reduce multicollinearity and cross validation to select the hyperparameter C. After that we fitted a decision tree using cross validation to select the hyperparameter ccp_alpha. Then we used vanilla bagging, which is a random forest model with max_features set to the total number of features in the training set. Then we used cross validation to select the max_features value with the highest accuracy for a random forest model. Finally, we fitted a linear SVM and RBF SVM. We then evaluated their performances and found that the random forest model had the highest accuracy and TPR, and the lowest FPR out of all our fitted models. This is the final model that we decided to send to the frontend for our web application. Lastly, we looked at the most important features for the random forest model as well as general correlation for the training set.

## mental_health/images
Contains image files for all images in the application.


## mental_health/app.py
This is the Streamlit application written in python with the Streamlit module. This file is hosted using Streamlit Sharing services. 

## mental_health/pages
This directory contains the files written in python using Streamlit API to display each of the pages for the application. The survey page displays the survey and records the responses into a database. The Home and About pages display information about our product. The Results page loads our ML solution, and inputs the survey responses from the database to achieve a prediction and displays an analysis of the results.

## braches
The master branch is the minimum viable product according to acheiving simple UI display and the new_feature branch is implementing upgrades to the UI and tieing in changes to the results analysis as they are developed.
