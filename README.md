Using Natural Language Processing and supervised learning to classify distress messages during catastrophic events.
---
**[Udacity Nanodegree in Data Science](https://www.udacity.com/course/data-scientist-nanodegree--nd025) - Project 3**


In this project I leverage skills in **data engineering**, **machine learning** and **web development** tools to build and deploy a web interactive application to categorize distress messages received during disasters.

To achieve this goal I used:
- **python3**:
  - Built extract, transform and the load (ETL) pipelines to process data to a sql database (__pandas, sqlalchemy__)
  - Used Natural Language Processing (__nltk, sklearn__) techniques to prepare unstructured text for modeling
  - Built the machine learning multi-target classifier (__sklearn__) to classify text data
- **HTML, Bootstrap4, JavaScript**: front-end webpage
- **Flask**: back-end
- **Heroku**: deployment from github


**📢Try it out here!**


If you have any questions or suggestions, just send me a quick message via [LinkedIn](https://www.linkedin.com/in/josecruz-phd/). **Enjoy!**


## Table of Contents

- [Introduction](#introduction)
- [Repository content](#repository-content)
- [The data](#the-data)
- [Requirements](#requirements)
- [Data pipelines](#data-pipelines)
- [The webapp](#the-webapp)
- [Licensing and Acknowledgements](#licensing-and-acknowledgements)


## Introduction

During times of disasters it is important to classify. 

This project includes an app that can be used by emergency workers to classify distress messages.


## Repository content

    - app
    | - template
    | |- master.html  # main page of web app
    | |- go.html      # classification result page of web app
    |- run.py         # Flask file that runs app

    - data
    |- disaster_categories.csv  # data to process
    |- disaster_messages.csv    # data to process
    |- process_data.py          # script to process data
    |- .db   # database to save clean data to

    - models
    |- train_classifier.py      # script to train, evaluate and save classifier
    |- classifier.pkl           # saved model

    - README.md
    - environment.yml


## The data

The dataset was provided by [Appen](https://appen.com/) and was made available by udacity content creators.

The `data/message.csv` contains distress messages, both in their original form and the corresponding English translation. The `data/categories.csv` contains the target categories used to classify the messages.

## Requirements

To use the python scripts on your local machine, you need to:

1. Install a python environment with jupyter notebooks (e.g., [anaconda distribution](https://www.anaconda.com/products/individual)).

2. Create an environment with the required packages by running on the anaconda shell:
```
conda env create -f environment.yml --name myenv
conda activate myenv
```

## Data Pipelines

To regenerate the SQL database with the clean data and retrain the classifier:

1. Download the repo to your local machine:

  ```
  git clone https://github.com/joseferncruz/udacity-ds-project3.git
  cd udacity-ds-project1/
  ```

2. Extract, Transform and Load the data into a SQL database:

  ```
  python process_data.py disaster_messages.csv disaster_categories.csv database.db
  ```
3. Use NPL and re-build the machine learning classifier:

  ```
  python train_classifier.py ../data/database.db classifier.pkl
  ```

## The webapp

You can access the deployed **[dashboard-app here](https://noisy-nyc-app.herokuapp.com/)** (internet connection required, loading may take around 2 minutes). Try it out!


## Licensing and Acknowledgements

The analysis and code generated during this project are licensed under a MIT License.

I want to thank [Udacity](https://www.udacity.com/) for providing the content of the project and [appen](https://appen.com/) for making these great data publicly available.


---
<h4 id="disclaimer">Disclaimer</h4>
 The author is not affiliated with any of the entities mentioned nor received any kind of compensation. The information contained in this work is provided on an "as is" basis with no guarantees of completeness, accuracy, usefulness or timeliness.
