#!/usr/bin/env python
"""
Train Classifier.

@author: Jose Oliveira da Cruz. 2021
"""
# coding: utf-8


# import libraries
import sys
import pickle
from sqlalchemy import create_engine
import pandas as pd
import re

# for npl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
STOP_WORDS_ENG = stopwords.words('english')

# for statistical learning
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split,\
                                    GridSearchCV
from sklearn.pipeline import Pipeline,\
                             FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer,\
                                            TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,\
                         TransformerMixin

random_seed = 10

# Classes to extract new features
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

###############################################################################

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


###############################################################################

def load_data(database_filepath):
    """Load data from a sql database.

    Parameters
    ----------
    database_filepath : str
        Filepath to the sql database.

    Returns
    -------
    X : np.ndarray

    y : pd.DataFrame

    category_names : np.ndarray

    """
    # Create engine to read sql database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # load data from database
    df = pd.read_sql_table('disaster_response', engine)

    # Extract documents from df
    X = df.loc[:, 'message'].values

    # Extract target categories
    y = df.iloc[:, 4:].astype('int')

    # Get target category names
    category_names = y.columns

    return X, y, category_names

###############################################################################

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
    tokens_lem = [token.strip().lower() for token in lemmatizer.lemmatize(tokens)
                  if token not in STOP_WORDS_ENG]

    return tokens_lem

###############################################################################

def build_model():
    """Build a classifier using sklearn pipelines and GridSearchCV.

    Returns
    -------
    model_cv : model

    """
    # Get the pipeline
    pipe = Pipeline(

    [('text_processing', FeatureUnion([('text_pipeline', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                                  ('tfidf', TfidfTransformer())])),
                                       ('verb_extract', StartingVerbExtractor()),
                                       ('n_tokens', GetNumberTokens())])
     ),
     ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=random_seed)))
     ]
     )

    # Perform Cross Validation to extract the best estimator
    # Set grid of parameters for cross validation
    params_grid = {
             'clf__estimator__n_estimators': [100, 200, 300], 
             'clf__estimator__learning_rate': [0.05, 0.35] 
    }

    # Build the model using gridsearch cross validation
    model_cv = GridSearchCV(pipe,
                            params_grid,
                            cv=3,
                            refit=True,
                            verbose=2,
                            n_jobs=-1,
                            return_train_score=True)

    return model_cv

###############################################################################

def evaluate_model(model, X_test, y_test, category_names):
    """Evaluate model and prints the classification report for each category.

    Parameters
    ----------
    model : sklearn model
    X_test : pd.DataFrame
    y_test : pd.DataFrame
    category_names : list
    """
    # Make a prediction
    y_pred = model.predict(X_test)

    # Print classification report for each target category
    print(classification_report(y_test, y_pred, target_names=category_names))



def save_model(model, model_filepath):
    """Saves model into a pickle file.

    Parameters
    ----------
    model : sklean model
    model_filepath : str
    """

    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))



def main():
    
    if len(sys.argv) == 3:

        # Parse command line filepaths
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')

        # Load data
        X, y, category_names = load_data(database_filepath)

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
