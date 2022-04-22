# Disaster Response Pipeline Project

### Introduction
This project is part of [Udacity](www.udacity.com) Data Science Nanodegree required assignments.

The goal is to build a pipeline that categorizes messages sent at the time of disaster events. This can be extremely useful for authorities to correctly filter data and pass in an efficient way to corresponding bodies that will be in charge of health and rescue tasks.

ETL and ML pipelines are built in two separate files, and a random forest model is trained using data gathered by Figure Eight (now [Appen](appen.com). Finally, the model is deployed to a web app showing some visualizations about the data, and providing a platform where any message can be categorized on the fly.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File description
- **app**: Folder.
  - **run.py**: Python file containing the backend code.
  - **templates**: Folder.
    - **go.html**: Html code for go page used to categorize any new message.
    - **master.html**: Html code for master page.

- **data**: Folder.
  - **disaster__categories.csv**: CSV file with category binary data for each message.
  - **disaster__messages.csv**: CSV file with messages translated to English, and in their original language.
  - **process__data.py**: Python file with ETL pipeline.
  - **DisasterResponse.db**: File to which the Database is loaded (created after following step 1 from Instructions).

- **models**: Folder.
  - **train__classifier.py**: Python file with ML pipeline.
  - **classifier.pkl**: Saved model (created after following step 1 from Instructions).

### How to interact?
For any questions you can contact me on dcalvomayo@gmail.com

### Licensing
You may take any code, but don't forget to cite the source. Take into account that some code was developed by [Udacity](www.udacity.com), and data was provided by Figure Eight (now [Appen](appen.com)).

