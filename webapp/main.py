#!/usr/bin/env python
# coding: utf-8
from flask import Flask

app = Flask(__name__)

import json
import plotly
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOP_WORDS_ENG = stopwords.words('english')


from flask import render_template, request#, jsonify
from plotly.graph_objs import Bar
import joblib


from sqlalchemy import create_engine

from sklearn.base import BaseEstimator,\
                         TransformerMixin


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """Return True if the first word is a Verb."""
        # Extract a list of tokenized sentences
        sentence_list = nltk.sent_tokenize(text)

        for sentence in sentence_list:

            # Calculate the number of tokens
            pos_tags = nltk.pos_tag(tokenize(sentence))

            if not pos_tags:
                return False

            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X_tagged = pd.Series(X).apply(self.starting_verb)

        return pd.DataFrame(X_tagged)


class GetNumberTokens(BaseEstimator, TransformerMixin):

    def get_number_tokens(self, text):
        """Return the number of tokens in a sentence."""
        # Extract a list of tokenized sentences
        n_tokens = len(tokenize(text))

        return n_tokens

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X_tagged = pd.Series(X).apply(self.get_number_tokens)

        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///./data/database.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Data
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    message_categories = df.iloc[:, 4:].sum(axis=0).sort_values(ascending=False).index
    number_elements = df.iloc[:, 4:].sum(axis=0).sort_values(ascending=False).values


    # Visuals
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],

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
        { 'data': [Bar(x=message_categories, y=number_elements)],

         'layout': {
             'title': 'Number of messages in each category',
             'yaxis': {
                 'title': "Count"
             },
             'xaxis': {
                 'tickangle': 45,
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

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
