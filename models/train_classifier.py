import sys

import numpy as np

import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

def load_data(database_filepath):
    """
    Function that loads message data from a SQLite database specified
    by its filepath.

    Input:
    - database_filepath: string with filepath where SQLite database is stored.

    Output:
    - X: Array containing all messages.
    - Y: pd.DataFrame containing categories binary data.
    - category_names: Array containing categories names.

    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterData',engine)
    Y = df.drop(['id','message','original','genre'], axis=1)
    X = df['message']
    category_names = Y.columns

    return X, Y, category_names

def tokenize(text):
    """

    Function that processes a string using tokenization and lemmatization
    techniques from nltk libraries.

    Input:
    - text: string containing text to be processed.
    
    Output:
    - tokens: Array containing processed data. 

    """

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    stop_words = stopwords.words('english')
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    tokens = [PorterStemmer().stem(word) for word in tokens]
    return tokens


def build_model():
    """

    Function that builds a ML pipeline to classify message data using sklearn
    library with the following algorithm:

    1. Convert tokenized messages into vectors.
    2. Apply TFID transformation to improve accuracy?
    3. Classify data using Random Forest technique.

    Output:
    - pipeline: pd.Pipeline containing the model.

    """

    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """

    Fucntion that evaluates a model on test data printing on stdout.

    Input:    
    - model: pd.Pipeline containing the model to evaluate.
    - X_test: Array containing messages test data.
    - Y_test: Dataframe containing categories binary test data.
    - category_names: Array with categories names.

    """

    Y_pred = model.predict(X_test)

    for i, column in enumerate(Y_test):
        print(column)
        print(classification_report(Y_test[column].values, Y_pred[:,i]))


def save_model(model, model_filepath):
    """

    Procedure that saves a model to a given filepath using library pickle.

    Input:
    - model: pd.Pipeline containing the model.
    - model_filepath: String with filepath where the model will be stored.

    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():

    # Check that all arguments are given
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # Split data into train and test. Modify test_size if needed.
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
