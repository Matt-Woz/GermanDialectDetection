import string
import re
import codecs
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics


def loadData():
    belgian_df = pd.read_csv('C:/Users/Matt/Documents/TYP/BelgischeDebatten/BelgischeDebatten1.txt', "utf-8",
                             header=None, names=["Belgian"])
    australian_df = pd.read_csv('C:/Users/Matt/Documents/TYP/Australiendeutsch/Australiendeutsch1.txt', "utf-8",
                                header=None, names=["Australian"])
    return belgian_df, australian_df


def createDataframe():
    belgian_df, australian_df = loadData()
    data_blg, lang_blg = preProcessing(belgian_df, "Belgian")
    data_aus, lang_aus = preProcessing(australian_df, "Australian")
    df = pd.DataFrame({"Text": data_blg + data_aus,
                       "language": lang_blg + lang_aus})
    splitdata(df)


def splitdata(df):
    X, y = df.iloc[:, 0], df.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)

    createPipeline(X_train, X_test, y_train, y_test)


def createPipeline(X_train, X_test, y_train, y_test):
    vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 3), analyzer='char') #Using trigrams

    pipeline_logisticRegression = pipeline.Pipeline([
        ('vectorizer', vectorizer),
        ('clf', linear_model.LogisticRegression())
    ])

    pipeline_logisticRegression.fit(X_train, y_train)

    y_predicted = pipeline_logisticRegression.predict(X_test)
    evaluateModel(y_test, y_predicted)


def evaluateModel(y_test, y_predicted):
    accuracy = (metrics.accuracy_score(y_test, y_predicted))*100
    print(accuracy,'%')

    matrix = metrics.confusion_matrix(y_test, y_predicted)
    print('Confusion matrix: \n', matrix)

def main():
    createDataframe()


def preProcessing(dataframe, language):
    data = []
    lang = []
    translate_table = dict((ord(char), None) for char in string.punctuation)
    for i, line in dataframe.iterrows():
        line = line[language]
        if len(line) != 0:
            if "Transkriptzeile" in line:
                continue
            line = line.lower()
            line = re.sub(r"\d+", "", line)
            line = re.sub(r"\t\w\w\t", "", line)
            line = line.translate(translate_table)
            data.append(line)
            lang.append(language)
    return data, lang


if __name__ == '__main__':
    main()
