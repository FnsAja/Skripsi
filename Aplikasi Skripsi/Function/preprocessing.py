import pandas as pd 
import numpy
import re
import json
import lib.tesaurus.tesaurus as ts
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, KFold
from sklearn import svm, metrics

def longWordRemoval(sentence):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1", sentence)

def synonymWordFilter(list):
    list_synonym = []
    filtered_list = []
    for word in list:
        if word not in list_synonym and word not in filtered_list:
            filtered_list.append(word)
        else:
            continue
        
        list_synonym.extend(ts.getSinonim(word))

    return filtered_list

def slangWordFilter(list):
    filtered_list = []
    
    for word in list:
        if word in list_slangwords.keys():
            filtered_list.append(list_slangwords[word])
        else:
            filtered_list.append(word)
    
    return filtered_list

data_source_url = "../Anies_JanFeb (Done).xlsx"
read_file = pd.read_excel(data_source_url)
read_file.to_csv("Data.csv", index=None, header=True)
tweets = pd.read_csv("Data.csv")

features = tweets.iloc[:, 1].values
labels = tweets.iloc[:, 0].values

processed_features = []

factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords = stopwords.words('indonesian')

new_stopwords = open('./lib/NLP_bahasa_resources/combined_stop_words.txt').read().split("\n")
list_stopwords.extend(new_stopwords)

list_slangwords = json.load(open('./lib/NLP_bahasa_resources/combined_slang_words.txt'))

print("Pre Processing")

for sentences in features:
    # Remove Special Character
    temporary = re.sub(r'\W', ' ', str(sentences))

    # Remove Number
    temporary = re.sub(r'\d', ' ', temporary)

    # Remove Single Character
    temporary = re.sub(r'\s+[a-zA-Z]\s+', ' ', temporary)

    # Remove Many Spaces
    temporary = re.sub(r'\s+', ' ', temporary.strip(), flags=re.I)

    # Lowering Case
    temporary = temporary.lower()

    # Long Word Removal ex : aamiiinn => amin
    temporary = longWordRemoval(temporary)

    # Stemming Bahasa Indonesia
    temporary = stemmer.stem(temporary)

    # Tokenize
    token_result = word_tokenize(temporary)
    
    # Remove Stop Words
    stopwords_result = [word for word in token_result if not word in list_stopwords]
    
    # Remove Slang Words
    slangwords_result = slangWordFilter(stopwords_result)
    
    # Synonym Words
    synonym = synonymWordFilter(slangwords_result)
    
    # Back To String
    temporary = ' '.join(map(str, synonym))
    
    # Remove Multiple Word
    temporary = ' '.join(set(temporary.split()))

    # Tambahkan kedalam list
    processed_features.append(temporary)

    print(f"Pre-Processing - {round(len(processed_features)/len(features)*100, 2)}%")

print("Processing")

vectorizer = TfidfVectorizer()
calculate_features = vectorizer.fit_transform(processed_features)
clf = svm.SVC(kernel="rbf")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# score = cross_val_score(clf, calculate_features, labels, cv=kf)

i = 1
for train_index, test_index in kf.split(calculate_features, labels):
    X_train = calculate_features[train_index]
    y_train = labels[train_index]
    X_test = calculate_features[test_index]
    y_test = labels[test_index]

    # Train
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    
    # Display Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
    confusion_matrix = numpy.flipud(confusion_matrix)
    confusion_matrix = numpy.fliplr(confusion_matrix)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Positive", "Netral", "Negative"])
    cm_display.plot()
    # plt.show()
    print(metrics.classification_report(y_test, y_predict))

    countNetral = 0
    countPositive = 0
    countNegative = 0

    falsePositive = 0
    falseNegative = 0
    falseNetral = 0
    truePositive = 0
    trueNegative = 0
    trueNetral = 0

    # for index, content in enumerate(y_predict):
        # Prediksi Salah
        # if content != labels[test_index][index]:        
        #     print(f"Fold {i} Isi {processed_features[test_index[index]]} Labels {content} Correct {labels[test_index[index]]}")
        
        # Jumlah Positif, Negatif dan Netral
        # if content == 0:
        #     countNetral += 1            
        # elif content == 1:
        #     countPositive += 1            
        # elif content == -1:
        #     countNegative += 1

        # Jumlah True Positive, False Positive, True Netral, False Netral, True Negative, False Negative
        # if content == 0:
        #     if content == labels[test_index[index]]:
        #         trueNetral += 1
        #     else:
        #         falseNetral += 1
        # elif content == 1:
        #     if content == labels[test_index[index]]:
        #         truePositive += 1
        #     else:
        #         falsePositive += 1
        # elif content == -1:
        #     if content == labels[test_index[index]]:
        #         trueNegative += 1
        #     else:
        #         falseNegative += 1

    # print(f"Fold {i} Positif {countPositive} Netral {countNetral} Negatif {countNegative}")
    
    # print(f"Fold {i} TrueP {truePositive} FalseP {falsePositive}")
    # print(f"Fold {i} TrueNt {trueNetral} FalseNt {falseNetral}")
    # print(f"Fold {i} TrueN {trueNegative} FalseN {falseNegative}")

    i += 1