import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function reads the csv files containing messages and categories into 
    dataframes,  then merges them 
    
    Input: messages_filepath - file path of disaster_messages.csv
           categories_filepath - file path of disaster_categories.csv
           
    Output: df - dataframe containing merged data 
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on = 'id')
      
    
    return df

def clean_data(df):
    '''
    Function that derives feature names and values from messages data, 
    then drops the duplicates found and returns the cleaned dataframe 
    
    Input: df - dataframe containing merged data 
           
    Output: df - cleaned dataframe
    
    '''    
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # extract list of new column names for categories and rename as apropriate
    row = categories.iloc[0]
    category_colnames  = [re.sub('-[0-9]', '', c) for c in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace('[a-z].+-','')

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
  
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, on=df.index)
    
    # convert to binary and drop duplicates
    df['related'] = df['related'].apply(lambda x: 0 if x==0 else 1)
    df = df.drop(['id', 'key_0'], axis = 1).drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Function that saves the cleaned dataframe to an SQLite database as
    a table 
    
    Input: df - cleaned dataframe 
           database_filename - name of database in which to store data 
           
    
    '''
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('clean_cats', engine, index=False, if_exists = 'replace')  


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