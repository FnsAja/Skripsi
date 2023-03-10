import pandas as pd 
import re
import json
import lib.tesaurus.tesaurus as ts
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
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

data_source_url = "./Datatest.csv"
tweets = pd.read_csv(data_source_url, delimiter=";")

features = tweets.iloc[:, 1].values
labels = tweets.iloc[:, 0].values

processed_features = []

factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords = stopwords.words('indonesian')

new_stopwords = open('./lib/NLP_bahasa_resources/combined_stop_words.txt').read().split("\n")
list_stopwords.extend(new_stopwords)

list_slangwords = json.load(open('./lib/NLP_bahasa_resources/combined_slang_words.txt'))


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

# review_dict = {'tweet': processed_features, 'sentimen' : labels}
# df = pd.DataFrame(review_dict, columns = ['tweet', 'sentimen'])

# vectorizer = TfidfVectorizer()
# processed_features = vectorizer.fit_transform(processed_features)
# clf = svm.SVC(kernel="rbf")
# kf = KFold(n_splits=10, shuffle=True, random_state=42)

# score = cross_val_score(clf, processed_features, labels, cv=10).mean()

# print(score)

# i = 1
# for train_index, test_index in kf.split(processed_features, labels):
    # clf.fit(processed_features[train_index], labels[train_index])
    # y_predict = clf.predict(processed_features[test_index])
    
    # print("-----------------------------------------------------------------------")
    # print(f"Accuracy - {i}: ", metrics.accuracy_score(labels[test_index], y_predict))
    # print(f"Precision - {i}: ", metrics.precision_score(labels[test_index], y_predict))
    # print(f"Recall - {i}: ", metrics.recall_score(labels[test_index], y_predict))
    # print(f"AUC - {i}: ", metrics.roc_auc_score(labels[test_index], y_predict))
    # print("-----------------------------------------------------------------------")
    
    # i += 1