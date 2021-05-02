# compassion.ai
Compassion.ai is a brand new machine learning tool that will help companies assess their current mental health climate and provide recommendations for the best ways to be more accomodating for their workers.

## MentalHealth-FinalCode
This notebook contains the code used to create the model to predict whether or not a company takes mental health as seriously as physical health. The notebook starts with cleaning the csv files, which are gathered from Open Sourcing Mental Illness, a non-profit, dedicated to raising awareness, educating, and providing resources to support mental wellness in the tech and open source communities. The data used come from three different csv files, for the years 2014, 2016, and 2019. These files can be found in the data folder. After cleaning the data, we built 7 different models: a baseline model, logistic regression, a decision tree, vanilla bagging, random forest, linear SVM, and RBF SVM. We then evaulated their performances and found that the logsitic regression model had the highest accuracy and TPR, and the lowest FPR out of all our fitted models. This is the final model that we decided to send to the frontend for our web application. Lastly, we looked at th emost important features for the model as well as general correlation for the training set.

All code for web application hosted by streamlit in app.py
csv files gathered from [this website].
