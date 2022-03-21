import time
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn import feature_extraction, dummy
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import get_data


def splitdata(df):
    global start
    start = time.time()
    X, y = df.iloc[:, 0], df.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0)  # Randomly splits data into training and testing set

    createModel(X_train, X_test, y_train, y_test)


def createModel(X_train, X_test, y_train, y_test):
    vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(4, 4), analyzer='char',
                                                         stop_words=stopwords.words(
                                                             'german'))  # Vectoriser with tf-idf statistic
    """
    pipeline_NB = pipeline.Pipeline([
        ('vectorizer', vectorizer),
        ('clf', naive_bayes.MultinomialNB())
    ])
    
    pipeline_SVM = pipeline.Pipeline([
        ('vectorizer', vectorizer),
        ('clf', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
    ])
        pipeline_baseline = pipeline.Pipeline([
        ('vectorizer', vectorizer),
        ('clf', dummy.DummyClassifier(strategy="most_frequent"))
    ])
    
    pipeline_RandomForest = pipeline.Pipeline([
        ('vectorizer', vectorizer),
        ('clf', RandomForestClassifier())
    ])
    """
    pipeline_SVM = Pipeline([
        ('vectorizer', vectorizer),
        ('sampling', SMOTE(random_state=27)),  # Example with SMOTE
        ('classification', svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto'))
    ])
    pipeline_SVM.fit(X_train, y_train)  # Fits the model

    y_predicted = pipeline_SVM.predict(X_test)  # Use fitted model on test data
    evaluateModel(y_test, y_predicted, pipeline_SVM)


# Reveals results of the model and creates confusion matrix
def evaluateModel(y_test, y_predicted, clf):
    print("Time taken: ", time.time() - start)
    print(classification_report(y_test, y_predicted))
    accuracy = (metrics.accuracy_score(y_test, y_predicted)) * 100
    print(accuracy, '%')
    matrix = metrics.confusion_matrix(y_test, y_predicted)
    print('Confusion matrix: \n', matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=clf.classes_)
    disp.plot()
    plt.xticks(rotation=45, ha='right')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()


#  Used to find total and unique word count of a corpus
def findWordCount(data):
    totalCounter = 0
    uniqueCounter = 0
    unique_words = []
    for i in range(len(data)):
        totalCounter = totalCounter + len(data[i].split())
        temp = data[i].split()
        for word in temp:
            if word not in unique_words:
                unique_words.append(word)
                uniqueCounter += 1
    print("total " + str(totalCounter))
    print("unique " + str(uniqueCounter))


#  Used to get all class combinations for dialect-distance comparison
def find_combinations(labels):
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or i < j:
                continue
            print(labels[i] + " - " + labels[j])


def main():
    group1, group2, group3, group4 = get_data.createDataframe(isHeatmap=False)
    splitdata(group4)


if __name__ == '__main__':
    main()
