import re
import string


def preProcessing(dataframe, language):
    data = []
    lang = []
    punctuation_dictionary = dict((ord(char), None) for char in string.punctuation)
    for i, line in dataframe.iterrows():
        line = line[language]
        if len(line) != 0:
            if "Transkriptzeile" in line:  # Removes remnant from reformatting corpora
                continue
            line = line.lower()
            line = re.sub(r"\d+", "", line)  # Removes digits e.g. line numbers
            line = re.sub(r"\t\w+\t", "", line)  # Removes initials at start of transcript
            line = re.sub(r"\t+", "", line)  # Removes unnecessary tabs
            line = line.translate(punctuation_dictionary)  # Removes punctuation
            if line in data:
                continue  # Ensures no duplicate sentences
            if len(line) > 1:
                data.append(line)
                lang.append(language)
    return data, lang
