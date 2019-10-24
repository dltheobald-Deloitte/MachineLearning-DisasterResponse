import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Reads in data from the 2 filepaths given into dataframes and merges them on 'id' to give one dataframe.
    
    Parameters:
    messages_filepath (string): A string containing the location of the messages data.
    categories_filepath (string): A string containing the location of the categories data.
    sep (string): The seperator which the column (col) mentioned above needs to be split over.

    Returns:
    df (pandas.DataFrame): A dataframe containing the merged data of both the messages and categories.
    
    """
    #Loading messages dataset
    messages = pd.read_csv(messages_filepath)

    #Loading categories dataset
    categories_filepath = pd.read_csv(categories_filepath)

    #merging together messages and categories dataset
    df_merged = messages.merge(categories_filepath, on = 'id')

    return df_merged


def clean_data(df):
    """ Cleans the input df by dropping suplicates and splitting the categories columns into multiple columns with corresponding headings and
    extracting the 1 or 0 from the end of the string, converting it into an integer. 
    
    Parameters:
    df (pandas.DataFrame): A dataframe containing the unclean merged data of both the messages and categories.

    Returns:
    df (pandas.DataFrame): A cleaned version of the input df, splitting categories data into individual columns.
    
    """
    #Splitting categories information into multiple columns
    categories = df['categories'].str.split(';', expand = True)
    
    #Extracting category column names and renaming columns in the 'categories' dataframe
    row = categories.iloc[0]
    category_colnames = [entry[:-2] for entry in row]
    categories.columns = category_colnames

    #Extracting the useful 1 / 0 endings of new categories columns, converting to integers.
    for column in categories.columns:
        #Keeping last character in each categories column entry.
        categories[column] = categories[column].apply(lambda x : x[-1:])
    
        # convert column from string to integer
        categories[column] = categories[column].astype('int')

    #Dropping original categories column, adding the new columns and removing duplicates
    df = df.drop(['categories'], axis = 1)
    df = pd.concat([df, categories], axis = 1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """ Saves the data stored in df into an sqlite database with the name {database_filename} and table name 'cleaned_massages'
    
    Parameters:
    df (pandas.DataFrame): A finalised dataframe containing the cleaned and merged data of both the messages and categories.
    database_filename (string): A string containing the database name where to save the data in df.
    
    """
    #Creating new sqlite database
    engine = create_engine('sqlite:///' + database_filename)

    #Uploading data to 'cleaned_messages' table
    df.to_sql('cleaned_messages', engine, index=False)  


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