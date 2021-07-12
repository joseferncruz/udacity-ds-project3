Using Natural Language Processing and supervised learning to classify distress messages during catastrophic events.
---
**[Udacity Nanodegree in Data Science](https://www.udacity.com/course/data-scientist-nanodegree--nd025) - Project 3**


In this project I leverage skills in **data engineering**, **machine learning** and **web development** tools to build and deploy a web interactive application to categorize distress messages received during disasters.

💡 To achieve this goal I used:
- ✅**python3**:
  - Built extract, transform and the load (ETL) pipelines to process data to a sql database (__pandas, sqlalchemy__)
  - Used Natural Language Processing (__nltk, sklearn__) techniques to prepare text for modeling
  - Built the machine learning multi-target classifier (__sklearn__) to classify text data
- ✅**HTML, Bootstrap4, JavaScript**: front-end webpage
- ✅**Flask**: back-end
- ✅**Heroku**: deployment from github

**🔴[Try it out here](https://classifying-disaster-messages.herokuapp.com/)👈**


If you have any **questions** or **suggestions**, just send me a 💬 via [**LinkedIn**](https://www.linkedin.com/in/josecruz-phd/). **Enjoy!**


## Table of Contents

- [Introduction](#introduction)
- [Repository content](#repository-content)
- [The data](#the-data)
- [Requirements](#requirements)
- [Data pipelines](#data-pipelines)
- [The webapp](#the-webapp)
- [Licensing and Acknowledgements](#licensing-and-acknowledgements)


## 📖 Introduction

During disaster events, thousands of distress messages are sent to organizations that can provide help. Unfortunately, during these periods, these organizations have the least capacity to screen all the messages and redirected them to specific units that can help and relief.

The purpose of this project is to build a machine learning model that can read messages and classify them into one of 36 different categories depending on their content. This would allow organization to better direct their resources and maximize assistance.


## 📂 Repository content

    - webapp
    | - template
    | |- master.html     # main page of web app
    | |- go.html         # classification result page of web app
    |- main.py         # Flask file that runs app
    |- __init__.py

    - data
    |- categories.csv     # data to process
    |- messages.csv       # data to process
    |- process_data.py    # script to process data
    |- database.db        # database to save clean data to

    - models
    |- train_classifier.py      # script to train, evaluate and save classifier
    |- classifier.pkl           # saved model

    - README.md
    - LICENSE
    - Procfile
    - .gitignore
    - classifying-disaster-messages.py
    - environment.yml
    - requirements.txt
    - nltk.text
    - runtime.txt
<br> </br>

## 📊 The data

The dataset was provided by [Appen](https://appen.com/) and was made available by Udacity content creators.

The `data/message.csv` contains distress messages, both in their original form and the corresponding English translation. The `data/categories.csv` contains the target categories used to classify the messages.

<br> </br>
## 🔨 Requirements

To use the python scripts on your local machine, you need to:

1. Install a python environment with jupyter notebooks (e.g., [anaconda distribution](https://www.anaconda.com/products/individual)).

2. Create an environment with the required packages by running on the anaconda shell:
```
conda env create -f environment.yml --name myenv
conda activate myenv
```

<br> </br>
## ⚠️ Data Pipelines

To regenerate the SQL database with the clean data and retrain the classifier:

1. Download the repo to your local machine:

  ```
  git clone https://github.com/joseferncruz/udacity-ds-project3.git
  cd udacity-ds-project1/
  ```

2. Extract, Transform and Load the data into a SQL database:

  ```
  python process_data.py messages.csv categories.csv database.db
  ```
3. Use NLP to process text and re-build the machine learning classifier:

  ```
  python train_classifier.py ../data/database.db classifier.pkl
  ```

<br> </br>
## 💻 The webapp

You can access the deployed **[dashboard-app here](https://classifying-disaster-messages.herokuapp.com/)** (internet connection required, loading may take around 2 minutes). __Try it out!__

<br> </br>
## 📃 Licensing and Acknowledgements

The analysis and code generated during this project are licensed under a MIT License.⃣

I want to thank [Udacity](https://www.udacity.com/) for providing the content of the project and [appen](https://appen.com/) for making these great data publicly available.


---
<h4 id="disclaimer">Disclaimer</h4>
 The author is not affiliated with any of the entities mentioned nor received any kind of compensation. The information contained in this work is provided on an "as is" basis with no guarantees of completeness, accuracy, usefulness or timeliness.
