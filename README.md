# disaster_classifier

### Project Motivation
The aim of this project was to develop a ML pipeline, featuring natural language processing , capable of classifying messages indicating disaster. 

### File Descriptions

process_data.py : reads in, processes and cleans the data. This file stores the cleaned data in DisasterResponse.db, as a table named clean_cats. 

train_classifier.py : lifts data from database and splits into input and target variables. A custom tokenisation function has been produced to normalise, tokenize and lemmatize the input text. This function is incorporated within the pipeline which vectorises the tokenized text, transforms to a tf - idf representation and then performs a multi-output classification using a KNeighbours classifier. A grid search is used to optimise the model parameters, before the model is trained and then saved as a pickle file. 

run.py : once the model is saved, a flask app can be run using this file, capable of classifying messages submitted. 

Directions to run the code and use the app can be found below. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Type env|grep WORK into the directory, then https://SPACEID-3001.SPACEDOMAIN, filling in SPACEID and SPACEDOMAIN as appropriate


### Licensing, Authors, Acknowledgements
Thanks to Udacity for FAQs and providing the template for the code

### Libraries Used
- pandas
- numpy
- re
- sqlalchemy
- pickle
- nltk
- sklearn
