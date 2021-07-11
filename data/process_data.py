#!/usr/bin/env python
"""
Process data.
@author: Jose Oliveira da Cruz. 2021
"""
# coding: utf-8

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

###############################################################################

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories into a single pandas dataframe.

    Parameters
    ----------
    messages_filepath : str
    
    categories_filepath : str

    Returns
    -------
    df : pd.DataFrame
        A dataframe containing the categories and messages.

    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df

###############################################################################

def clean_data(df):
    """Preprocess and clean data.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe containing the categories and messages.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with preprocessed, clean messages/categories data.

    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0].str.replace(r'-\d', '', regex=True).values
    category_colnames = row.copy()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(r'\D',
                                                            '',
                                                            regex=True)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # normalize pos class for the feature related
    categories['related'].replace(2, 0, inplace=True)

    # Remove group with only one value (ie 0)
    categories.drop("child_alone", axis=1, inplace=True)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)


    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    print(f'Found {df.duplicated().sum()} duplicate records.',
          'Removing duplicates...')

    # drop duplicates
    df.drop_duplicates(inplace=True)
    print(f'Removed duplicates. {df.duplicated().sum()} duplicates found.')

    return df

###############################################################################


def save_data(df, database_filepath):
    """Save dataframe into a local sql database.

    Parameters
    ----------
    df : pd.DataFrame
    
    database_filepath : str
    
    """
    # save output in a sql database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    df.to_sql('disaster_response', engine, index=False)


###############################################################################

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        # print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

###############################################################################

if __name__ == '__main__':
    main()
