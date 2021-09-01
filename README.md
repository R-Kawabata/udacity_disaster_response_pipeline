# Disaster Response Pipeline Project

### Introduction

This is one of Udacity Data Science Nanogreee project. The goal of this project is to deploy a web application, which receives a disaster message and classify it according to its category.

In order to reach the objective, it was necessary to:
- build a ETL pipeline to extract the data from the [disaster data set](https://www.figure-eight.com/dataset/combined-disaster-response-data/) and store the clean data to a SQLite Database
- build a Machine Learning pipeline, using libraries such as scikit-learn, NLTK and sqlachemy. The data from ETL pipeline is tokenized and splited to tarinig and testing data. After ML model evaluation, the model was stored to be used later by web application.

### Libraries
- pandas 1.2.4
- numpy 1.20.1
- sqlalchemy 1.4.7
- pickle 4.0
- re 2.2.1
- nltk 3.6.1
- scikit-learn 0.24.2


### Files in the repository
```app``` a folder containing the run.py and a templete for the web application

```data/DisasterResponse.db``` the databese generated by ETL pipeline in the process_data.py

```data/disaster_categories.csv``` a data set containing the categories of the disaster messages

```data/disaster_messages.csv``` a data set containing the disaster messages

```data/process_data.py``` ETL pipeline. It cleans, merges and stores the data set as SQLite database.

```models/train_classifier.py``` ML pipeline. It loads the SQLite and generates the data to traing and test the ML model. It also build the model using the custom parameters and shows the precision, recall, f1-score and accuracy of the model. At the end, it saves the model as pickle file.

### Data set source
[Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results
![](img_samples/web_application_sample.JPG)
