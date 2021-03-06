import json
import plotly
import pandas as pd
import numpy as np
import sys, os

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

sys.path.append('../models/')

from train_classifier import tokenize

app = Flask(__name__)

# Function commented out because it is directly imported from ../models/ 
# for consistency

#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()
#
#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)
#
#    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    #genre_counts = list(genre_counts)

    # Obtain categories names and counts
    cat_names = df.columns[4:]
    cat_counts = df[df.columns[4:]].sum()

    # Obtain messages lengths
    lengths = np.zeros(len(df.index))
    for i, ind in enumerate(df.index):
        lengths[i] = int(len(df['message'][ind]))

    # Compute histogram with bins spaced logarithmically
    bins = np.logspace(1,4,num=50)
    len_freqs, len_bins = np.histogram(lengths, bins=bins)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }    
        },

        # Bar chart for number of messages on each category 
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Number of messages on each category',
                'yaxis': {
                    'title': "Count"
                }
            }
        },

        # Histogram of message length
        {
            'data': [
                Bar(
                    x = len_bins,
                    y = len_freqs,
                    width = np.diff(bins)
                )
            ],

            'layout': {
                'title': 'Distribution of Message Length',
                'align': 'edge',
                'xaxis': {
                    'type': 'log',
                    'title': 'Number of characters'
                    },
                'yaxis': {
                    'title': 'Frequency'
                    }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
