import json
import plotly
import pandas as pd
import re

import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly import graph_objs as go
from plotly.graph_objs import Bar, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys
sys.path.append('..')
from models.train_classifier import text_length, tokenize


app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cleaned_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def make_graph(data, title, x_label, y_label):
    """make_graph creates a dictionary/json-format version of a graph which can be used to
    generate a graph in html.

    Parameters:
    data (Plotly GraphObject): data containing x and y points in graph object format
    title (String): The total desired on the graph on the webpage
    x_label (String): The label desired on the x axis
    y_label (String): THe label desired on the y axis

    Returns:
    graph (dict): A dictionary containing the data and labels neccessary to generate a graph
    """
    graph = {'data': [data],
            'layout': {
                'title': title,
                'yaxis': {'title': x_label},
                'xaxis': {'title': y_label}
            }}
    
    return graph

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Renders a html template for the home page ('index'). This method also generates Plotly objects
    and manipulates the data to displays the graphs on the homepage.

    Returns:
    Returns a rendered template for the homepage for the web_app.
    """
    #Instatiating graph list to pass to html
    graphs = []

    #Counting the number of messages for each genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Generating a Bar Graph object and appending to graphs
    graph_1 = make_graph(Bar(x=genre_names,y=genre_counts),
                        'Distribution of Message Genres',
                        'Count', 'Genre')

    graphs.append(graph_1)

    #Melting the original dataframe to extract categry information, keeping only message and genre columns
    df_melt = pd.melt(df[list(df.columns[3:])+['message']],
                        id_vars = ['genre','message'],
                        var_name = 'categories',
                        value_name = 'Flag' )
    
    #Generating 3 bar graphs traces, one for each genre
    data_test = []
    for genre in genre_names:
        categories_sum = df_melt[df_melt['genre'] == genre].groupby('categories')['Flag'].sum()
        categories = list(categories_sum.index)

        trace = Bar(name = genre, x=categories, y=categories_sum)
        data_test.append(trace)

    #Making graph object to plot the 3 traces above, appending to graphs
    graph_2 = Figure(data = data_test)
    graph_2.update_layout(barmode = 'group', title ='Number of tags for each Category and Genre',
                            xaxis = {'title': 'Categories'}, yaxis = {'title' : 'No. of Tags'})

    graphs.append(graph_2)

    #Calculating the message length
    df_melt['message_length'] = df_melt['message'].apply(lambda x : len(x))

    #Taking only the assigned labels (i.e FLag > 0) and getting the mean message length for each
    length_by_category = df_melt[df_melt['Flag'] > 0].groupby('categories')['message_length'].mean()
    category_names = list(length_by_category.index)

    #Making Bar graph object and append to graphs
    graph_3 = make_graph(Bar(x=category_names,y=length_by_category),
                    'Average Message Length by Category',
                    'Message Length', '')             
    
    graphs.append(graph_3)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Renders a html template to classify a message into categories ('go'). 
    This method also takes in a message and classifies it for the above.

    Returns:
    Returns a rendered template for the 'go' page for the web_app.
    """
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


def main():
    """When this script is run as the main script, it runs a flask app on:
        http://127.0.0.1:5000
    """
    app.run(host = '127.0.0.1', port =5000, debug = True)


if __name__ == '__main__':
    main()