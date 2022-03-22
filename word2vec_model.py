import numpy as np
import pandas as pd
import time
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn import metrics
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pathlib
import get_data
import nltk


def createword2VecModel(df):
    global start
    start = time.time()
    word2vec_model_file = str(pathlib.Path(__file__).parent) + r'\word2vec_' + '.model'  # Creates model file
    df['Tokenized_text'] = df['Text'].apply(stop_word_removal)
    df['Tokenized_text'] = [[nltk.word_tokenize(line)] for line in df['Tokenized_text']]  # Tokenizes
    X_train, X_test, y_train, y_test = splitdataWord2Vec(df)
    tokenized_text = pd.Series(df['Tokenized_text']).values
    model = Word2Vec(tokenized_text, min_count=1, vector_size=1000, workers=3, window=3, sg=1)  # creates w2v model
    model.save(word2vec_model_file)  # Saves model for re-use using gensim
    generateWord2Vectors(X_train, y_train, X_test, y_test, 1000)


def stop_word_removal(x):
    german_stop_words = stopwords.words('german')
    token = x.split()
    return ' '.join([w for w in token if not w in german_stop_words])


def splitdataWord2Vec(df):
    X, y = df.iloc[:, 2], df.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)
    X_train = X_train.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.to_frame()
    y_train = y_train.reset_index()
    y_test = y_test.to_frame()
    y_test = y_test.reset_index()
    return X_train, X_test, y_train, y_test


def generateWord2Vectors(X_train, y_train, X_test, Y_test, vector_size):
    first = True
    w2v_filename = str(pathlib.Path(__file__).parent) + r'\word_vectors.csv'
    w2v_model = Word2Vec.load(
        str(pathlib.Path(__file__).parent) + r'\word2vec_' + '.model')  # Load the created model
    with open(w2v_filename, 'w+') as word2vec_vectors:
        for _, row in X_train.iterrows():
            mean_vector = (np.mean([w2v_model.wv[word] for word in row['Tokenized_text']], axis=0)).tolist()  # Finds mean vector of each word in a sentence
            if first:
                column_headings = ",".join(str(column) for column in range(vector_size))  # Creates numbered columns
                word2vec_vectors.write(column_headings + "\n")
                first = False
            if type(mean_vector) is not list:  # mean_vector is list if it exists, if not, add list of 0s
                line = ",".join([str(0) for _ in range(vector_size)])
            else:
                line = ",".join([str(vector_element) for vector_element in mean_vector])  # If exists, concatenate file with mean vectors of sentence
            word2vec_vectors.write(line + "\n")
    trainWord2VecModel(y_train, X_test, w2v_model, Y_test, vector_size)


def trainWord2VecModel(y_train, X_test, model, Y_test, vector_size):
    word2vec_df = pd.read_csv(str(pathlib.Path(__file__).parent) + r'\word_vectors.csv')
    # SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    # SVM.fit(word2vec_df, y_train['language'])
    RF = RandomForestClassifier()
    RF.fit(word2vec_df, y_train['language'])
    testModel(X_test, model, Y_test, RF, vector_size)


def testModel(X_test, model, Y_test, fitted_algo, vector_size):
    test_word2vec = []
    for _, row in X_test.iterrows():
        mean_vector = np.mean([model.wv[token] for token in row['Tokenized_text']], axis=0).tolist()  # Finds mean vector of sentence in testing set
        if type(mean_vector) is not list:
            test_word2vec.append(np.array([0 for _ in range(vector_size)]))
        else:
            test_word2vec.append(mean_vector)
    test_predictions = fitted_algo.predict(test_word2vec)  # Uses model to predict testing set class
    evaluateModel(Y_test['language'], test_predictions, fitted_algo)


def evaluateModel(y_test, y_predicted, fitted_algo):
    print("Time taken: ", time.time() - start)
    print(classification_report(y_test, y_predicted))
    accuracy = (metrics.accuracy_score(y_test, y_predicted)) * 100
    print(accuracy, '%')
    matrix = metrics.confusion_matrix(y_test, y_predicted)
    print('Confusion matrix: \n', matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=fitted_algo.classes_)
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()


def main():
    group1, group2, group3, group4 = get_data.createDataframe(isHeatmap=False)
    createword2VecModel(group1)


if __name__ == '__main__':
    main()
