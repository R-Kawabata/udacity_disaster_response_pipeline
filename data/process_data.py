import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """

    description:
    receives 2 data sets, message and categories,
    and merge them to one data set.
    
    Inputs:
    messages_filepath: a csv file path containing the message data set
    categories_filepath: a csv file path containing the category data set
    
    Output:
    a merged data set using the input data set
   
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge the data sets to one data set
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """

    description:
    Data cleaning function

    Inputs:
    df: a merged data frame (message + category)

    Output:
    cleaned data frame

    """    
    # get right categories column name
    categories = df['categories'].str.split(';', expand=True)
    row = categories.loc[0, :]
    category_colnames = [(lambda row : category.split('-')[0])(category) for label, category in row.iteritems()]
    
    # set new column names
    categories.columns = category_colnames
    
    # get categories columns values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [(lambda categories : value_str.split('-')[1])(value_str) \
                              for label, value_str in categories[column].iteritems()]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicated data
    df.drop_duplicates(subset=['message', 'original'], inplace=True)
    
    # remove value 2 from related column
    df = df.loc[df['related'] != 2]
    
    return df
    
def save_data(df, database_filename):
    """
    
    description:
    it saves the data set to sql database file
    
    Inputs:
    df: a merged data frame (message + category)
    database_filename: a name for the databse file
    
    output:
    creates a sql database file, containing the data from
    input data set

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


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