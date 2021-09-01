import re
import sys
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    
    description:
    it reads a SQL database and load it to a pandas data frame.
    After that, it extract the X and Y data to train and test the
    ML model. Also, it gets the column names.

    input:
    database_filepath: a file path for the database file

    output:
    X and Y: the data for ML model
    category_list: a list of column names

    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query('SELECT * FROM DisasterResponse', engine)
    X = df['message']
    Y = df[df.columns[4:]]
    category_list = list(Y.columns)
    return X, Y, category_list


def tokenize(text):
    """

    description:
    clean, tokenize and lemmatize the input text string

    input:
    text: inpput text

    output:
    clear_tokens: list of tokens

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_list = re.findall(url_regex, text)
    
    for url in url_list:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clear_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clear_tokens.append(clean_tok)
    
    return clear_tokens


def build_model():
    """

    description:
    it creates a ml pipeline, and set parameters and
    a model to be used later

    output:
    Machine learning model

    """
    pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
    
    parameters = {
        'vect__max_df': [1],
        'tfidf__use_idf': [True],
        'clf__estimator__n_jobs':[-1],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    
    description:

    input:
    model: ML model
    X_test and Y_test: test data
    category_names: list of class names

    output:
    precision, recall, f1-score and accuracy for each class,
    and an average accuracy of the ML model

    """
    y_pred = model.predict(X_test)
    average_accuracy = 0
    print(classification_report(Y_test, y_pred, target_names = category_names))
    for i in range(len(category_names)):
        class_acc = accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])
        average_accuracy += class_acc
        print('category {}: {:.3f}'.format(category_names[i], class_acc))
    print('\n')
    print('average accuracy: {}'.format(average_accuracy/len(category_names)))


def save_model(model, model_filepath):
    """
    
    description:
    it saves the trained ML model

    input:
    model: ML model
    model_filepath: file path and pikles file name to save the ml model
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()