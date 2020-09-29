import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine 
import pickle

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    '''
    Function reads  the table from the SQLite database created previously into a dataframe
    and splits into input and target variables. The names of the target variables are also extracted. 
    
    Input: database_filepath - name of the database in which table is stored
    
    Output: X- input variables, 
            Y- target variables, 
            cat_names - list of target variable names
    
    '''
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table('clean_cats', engine)  
    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis = 1)
    cat_names = list(Y.columns) 
    
    return X, Y, cat_names 

def tokenize(text):
    '''
    Function that normalizes, tokenises and then lemmatizes text  
    
    Input: text- message as raw text
    
    Output: text_lm - tokenized  text 
    
    '''
    # Remove special charcaters and convert to lower case, then tokenize    
    text_tk = re.sub(r"[^a-zA-Z]", " ", text).lower()
    text_tk = word_tokenize(text_tk)
    
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize text to base verbs 
    text_lm = [lemmatizer.lemmatize(t, pos='v') for t in text_tk]
       
    return text_lm


def build_model():
    '''
    Function that builds pipeline, finds optimal solution from the parameters specified,
    then returns optimised model, on which the training data is fitted 
    
    
    Output: cv - model optimised according to grid search 
    
    '''
    
    # Build pipeline, using K neighbours method for classification 
    pipeline_KN = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                          ('tfidf', TfidfTransformer()), 
                          ('clf', MultiOutputClassifier(KNeighborsClassifier()))
                           ])
    
    # Parameters for grid search
    parameters = {'clf__estimator__n_neighbors': [3, 5, 7, 9], 
              'vect__ngram_range':  [(1, 1) , (1,2) ],
              #'vect__max_df': (0.5, 0.75, 1.0),
              #'vect__max_features': (None, 5000, 10000, 50000),
              #'clf__estimator__weights': ['uniform', 'distance']
                 }
    
    cv = GridSearchCV(pipeline_KN , param_grid = parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that uses optimised and fitted model to predict target variables, then 
    evaluates performance from known values, across all categories used for classification 
    
    Input: model- optimised model, fitted to training data 
           X_test - test input variables 
           Y_test - test target variables 
           category_names -  target variable names derived earlier 

    
     '''
    # Optimal parameters found according to grid search 
    print("Best Parameters:", model.best_params_)
    
    # predict on test data
    y_pred = model.predict(X_test)

     # display results across all categories
    for i in range (Y_test.shape[1]):
         print(classification_report(Y_test.iloc[:,i], pd.DataFrame(y_pred)[i]))

        
def save_model(model, model_filepath):
    '''
    Function that saves the optimised, fitted model as a pickle file 
    with the name specified
    
    Input: model- optimised model, fitted to training data 

    
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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