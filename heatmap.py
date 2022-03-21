import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pandas as pd
import get_data


def createHeatmap(data):
    newArray = np.ndarray((12, 12))
    for i in range(12):
        finalData = []
        for x in range(12):
            finalData.append(findSimilarity(str(data[i]), str(data[x])))
        newArray[i] = finalData
    labels = ['Australian', 'Berlin', 'Namibia', 'East_germany', 'Turkey', 'GB', 'Prussia', 'German_in_israel',
              'Viennese_in_israel', 'Eastern_territories', 'Russian_dialects', 'Hamburg']
    df = pd.DataFrame(newArray, columns=labels)
    plt.figure(figsize=(10, 10))
    sns.heatmap(df, yticklabels=labels, xticklabels=labels, annot=True, cbar_kws={'label': 'Shared vocab 0-1'})
    plt.xticks(rotation=45, ha='right')
    plt.gcf().subplots_adjust(bottom=0.3, left=0.3)
    plt.show()


#  Used to calculate vocabulary similarity between two dialects for the heatmap
def findSimilarity(textA, textB):
    textA = set(nltk.word_tokenize(textA))
    textB = set(nltk.word_tokenize(textB))
    intersection = len(textA.intersection(textB))
    difference = len(textA.symmetric_difference(textB))
    return round(intersection / float(intersection + difference), 2)


def main():
    data = get_data.createDataframe(isHeatmap=True)
    createHeatmap(data)


if __name__ == '__main__':
    main()
