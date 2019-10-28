# Disaster Response Pipeline Project

### Requirements:
The python libraries required by this project are contained within 'requirements.txt'

### Summary:
In this project, there is an ETL pipeline and ML pipeline used to clean data and train & optimise a model
which can classify a message into multiple categories.
This project also contains code for a Flask app which, once the data has been loaded and model trained, can use the above to classify text inputs into multiple categories. On the flask app homepage, there are also some data visualisations for the data used to train the model.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000

### Files:
Within the data folder, there is a script 'process_data.py' which can be used to clean the 2 csv files within the same folder. The cleaned data will then be saved in the data folder under 'DisasterResponse.db'.

Within the models folder, there is a script 'train_classifier.py' which can be used to load the cleaned data mentioned above, build a model, optimise it over a set of paramaeters and evaluate it's performance. This script will also save a copy of this model as 'classifier.pkl' within the same filepath.

Within the app folder, this contains all of the html templates for the webpage within the 'templates' folder. The 'run.py' script will start up the flask app which can be accessible in a web browser specified above. When a task is performed on the webpage, the functions within this script will be used to generate data visiualisations and classify text.