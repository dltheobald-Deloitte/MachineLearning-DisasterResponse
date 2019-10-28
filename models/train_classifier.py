import sys
import pandas as pd
import re
import pickle

from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Downloading the required nltk packages
nltk.download(['punkt','wordnet','stopwords'])


def load_data(database_filepath):
    """ Reads in data from a pre-populated sqlite database, splitting them into two sets:
        - one set of data to be used for feature engineering
        - one set of data containing the labels for each row.
   
    Parameters:
    database_filepath (string): A string containing the location of the database.

    Returns:
    X (pandas.DataFrame): A dataframe containing the data before features engineering to train from.
    y (pandas.DataFrame): A dataframe containing the labels data to train to.
    category_names (list of String): A list of the categories/labels we are trying to predict.  
    """
    #Creating a connection to the sqlite database
    engine = create_engine('sqlite:///' + database_filepath)

    #Reading in data from the cleaned_messages table
    df = pd.read_sql( "SELECT * FROM cleaned_messages", con = engine)

    #Allocating data to be used forfeatures or labels
    X = df['message']
    y = df[list(df.columns[4:])]

    #Providing category names for the labels
    category_names = df.columns[4:]

    return X, y, category_names


def tokenize(text):
    """ Cleans text by removing stopwords and punctuation, making text lowercase and outputs it into
    a useable format by lemmatizing words.
   
    Parameters:
    text (string): A string containing a full disaster response message.

    Returns:
    clean_words (list of String): A list containing lemmatized versions of cleaned words in the text, removign stopwords.  
    """
    #Replacing all non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-z0-9]",' ',text)
    
    #Splitting string into individual words, making them lowercase
    tokens = word_tokenize(text.lower())
    
    #Instatiated objects for processing
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    clean_words = []

    #Lemmatizing each word then removing stopwords
    for word in tokens:
        if word not in stop_words:
            token = lemmatizer.lemmatize(word)
            clean_words.append(token)

    return clean_words


def text_length(messages):
    """ Transforms the messages in the dataframe into the number of characters in the string
   
    Parameters:
    messages (pd.DataFrame): A DataFrame of the messages used to train the model

    Returns:
    frame (pd.DataFrame): A DataFrame of the lenght of messages used to train the model
    """
    X_data = pd.Series(messages).apply(lambda x: len(x)).astype('int')
    return pd.DataFrame(X_data)


def build_model():
    """ Creates an instance of model along with a machine learning pipeline which, when fit,
    will process the features data and select the best parameters from those given to improve the model outputs.

    Returns:
    cv (GridSearchCV object): A machine learning pipeline and parameters to configure a model and process
                              its features 
    """
    #Defines pipeline to calculate the tfidf of each message and pass through the categories features.
    #This pipeline will also trains the classifier
    pipeline = Pipeline([ 
    ('features',FeatureUnion([
        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('text_length', FunctionTransformer(text_length, validate=False))
    ])),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #A selection of parameters to optimise the model over
    parameters = {
    'clf__estimator__n_estimators' : [10, 30, 50],
    'clf__estimator__min_samples_split' : [2, 10] 
    }

    #Creates instance of model with the mentioned pipeline
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates and returns the models performance with respect to precision, recall and f1_score

    Parameters:
    model (GridSearchCV object): A fitted model which can be used to predictr results 
    X_test (pd.DataFrame): A dataframe with the test set of data to transform into features.
    Y_test (pd.DataFrame): A dataframe with the test set of data labels.
    category_names (list of String): A list with the category/label description
    """
    #Prediction outputs from fitted model
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns = category_names)
    
    #Comparing predictions to actuals and printing metrics
    for category in category_names:
        print(category)
        print(classification_report(Y_test[category], Y_pred_df[category]))


def save_model(model, model_filepath):
    """ Saves a trained version of the model in the location specified in model_filepath

    Parameters:
    model (GridSearchCV object): A fitted model which can be used to predict results 
    model_filepath (String): The location of where the model shoule be saved
    """
    #Saving a copy of the fitted model
    pickle.dump(model, open(model_filepath, 'wb+'))


def main():
    """ When this script is run as main, this creates a fitted model which can be used for precictions.
    It takes in specified system variables are used to exectue the machine learning pipeline:
        - loading in data
        - Builing the model
        - Evaluating the model
        - Saving the model
    """
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