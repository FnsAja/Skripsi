import pandas as pd 
import re
import tesaurus.tesaurus as ts
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn import svm

def wordElongationFilter(sentence):
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

data_source_url = "./Datatest.csv"
tweets = pd.read_csv(data_source_url, delimiter=";")

features = tweets.iloc[:, 1].values
labels = tweets.iloc[:, 0].values

processed_features = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = stopwords.words('indonesian')
new_stopwords = open('./NLP_bahasa_resources/combined_stop_words.txt').read().split("\n")
list_stopwords.extend(new_stopwords)
# list_slangwords = 

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

    # Word Elongation Removal
    temporary = wordElongationFilter(temporary)

    # Stemming Bahasa Indonesia
    temporary = stemmer.stem(temporary)

    # Tokenize
    token_result = word_tokenize(temporary)
    
    # Remove Stop Words
    stopwords_result = [word for word in token_result if not word in list_stopwords]
    
    # Synonym Words
    synonym = synonymWordFilter(stopwords_result)
    
    # Back To String
    temporary = ' '.join(map(str, synonym))

    # Tambahkan kedalam list
    processed_features.append(temporary)

vectorizer = TfidfVectorizer()
processed_features = vectorizer.fit_transform(processed_features)
# X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=0.3, random_state=0)
# train_processed_features = vectorizer.fit_transform(X_train)
# test_processed_features = vectorizer.transform(X_test)


# Train Model using fit()
# clf.fit(train_processed_features, y_train)

# # Predict
# y_pred = clf.predict(test_processed_features)

clf = svm.SVC(kernel="rbf")
kv = KFold(n_splits=10, random_state=40, shuffle=True)

# for index, res in enumerate(result):
#     print("Fold ke-" + str(index+1) + " : Accuracy(" + str(res['accuracy']) + "), Recall(" + str(res['recall']) + "), Precision(" + str(res['precision']) + "), F1 Score(" + str(res['f1_score']) + ")")
