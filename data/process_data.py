import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Function that loads the message and categories into
    a Dataframe.
    
    Input:
        - messages_filepath: Filepath to messages csv file.
        - categories_filepath: Filepath to categories csv file.
    
    Output:
        - df: Dataframe containing messages with their respective
        categories.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')

    return df


def clean_data(df):

    """ 
    Function that cleans the dataframe created by load_data.

    Input:
        - df: Dataframe created by load_data.
    
    Output:
        - df: Clean dataframe containing a column for each category.
    """

    categories = df['categories'].str.split(';', expand=True)
    # Select first row of categories and extract a list of new
    # column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames.values

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        # drop rows that have any value different from 0 or 1
        categories = categories.drop(categories[
                (categories[column]!=0) & (categories[column]!=1)
                ].index)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], sort=False, axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):

    """
    Function that saves dataframe obtained by clean_data to
    a SQlite database.

    Input:
        - df: Dataframe from clean_data.
        - database_filename: Filename of SQlite database to which
                             is saved.
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exist='replace')  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
