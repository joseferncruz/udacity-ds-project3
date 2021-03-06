#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request
app = Flask(__name__)

# https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules/58740659#58740659
import pickle
class CustomUnpickler(pickle.Unpickler):
    """Costum class to unpickle models."""

    def find_class(self, module, name):
        """Find class in unpickled model."""
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)

import json
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOP_WORDS_ENG = stopwords.words('english')

import plotly
from plotly.graph_objs import Bar

from sqlalchemy import create_engine

from sklearn.base import BaseEstimator,\
                         TransformerMixin

STOP_WORDS_ENG = stopwords.words('english')

def tokenize(text):
    """Case normalize, clean, tokenize and lemmatize text.

    Parameters
    ----------
    text : str

    Returns
    -------
    tokens_lem : list
        List of clean, normalized and lemmatized tokens.

    """
    # Remove non-alphanumeric characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)

    # tokenization
    tokens = word_tokenize(text)

    # lemmanitization
    lemmatizer = WordNetLemmatizer()
    tokens_lem = [lemmatizer.lemmatize(token.strip().lower()) for token in tokens
                  if token not in STOP_WORDS_ENG]

    return tokens_lem


# Classes to extract new features
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Transformer class to extract the starting verb in text."""

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
        """Fit the data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
        y : optional, array-like of shape (n_samples,) or (n_samples, n_targets)
            
        Returns
        -------
        self : returns an instance of self.
        """
        return self

    def transform(self, X):
        """Transform the data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
    
        Returns
        -------
        X_tagged : pd.DataFrame
            Transformed data.     
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        return pd.DataFrame(X_tagged)

class GetNumberTokens(BaseEstimator, TransformerMixin):
    """Transformer class to count the number of tokens in text."""
    
    def get_number_tokens(self, text):
        """Return the number of tokens in a sentence.
        
        Parameters
        ----------
        text : str
            
        Returns
        -------
        n_tokens : int
            The number of tokens in text.
        """
        # Extract a list of tokenized sentences
        n_tokens = len(tokenize(text))

        return n_tokens

    def fit(self, X, y=None):
        """Fit the data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            
        y : optional, array-like of shape (n_samples,) or (n_samples, n_targets)
            
        Returns
        -------
        self : returns an instance of self.
        """        
        return self

    def transform(self, X):
        """Transform the data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
    
        Returns
        -------
        X_tagged : pd.DataFrame
            Transformed data.     
        """        
        X_tagged = pd.Series(X).apply(self.get_number_tokens)
        
        return pd.DataFrame(X_tagged)

# load data
engine = create_engine('sqlite:///./data/database.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = CustomUnpickler(open('./models/classifier.pkl', 'rb')).load()

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
