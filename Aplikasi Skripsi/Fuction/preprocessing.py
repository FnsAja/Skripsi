import pandas as pd 
import re
import "https://github.com/victoriasovereigne/tesaurus.git"
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import svm, metrics

data_source_url = "./Datatest.csv"
tweets = pd.read_csv(data_source_url, delimiter=";")

features = tweets.iloc[:, 1].values
labels = tweets.iloc[:, 0].values

processed_features = []
factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

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

    # Stemming Bahasa Indonesia
    temporary = stemmer.stem(temporary)

    # Tokenize
    token_result = word_tokenize(temporary)
    
    # Remove Stop Words
    stopwords_result = [word for word in token_result if not word in list_stopwords]
    print("Before : ", stopwords_result)
    
    # Synonym Words
    synonym = map(tesaurus.getSinonim(), stopwords_result)
    print("Sinonim : ", synonym)
    
    # Back To String
    temporary = ' '.join(map(str, stopwords_result))

    # Tambahkan kedalam list
    processed_features.append(temporary)

# vectorizer = TfidfVectorizer()
# for i in range(1, 9):
#     X_train, X_test, y_train, y_test = train_test_split(processed_features, labels, test_size=i * 0.1, random_state=0)
#     train_processed_features = vectorizer.fit_transform(X_train)
#     test_processed_features = vectorizer.transform(X_test)

#     clf = svm.SVC(kernel="rbf")

#     # Train Model using fit()
#     clf.fit(train_processed_features, y_train)

#     # Predict
#     y_pred = clf.predict(test_processed_features)

#     print("Accuracy K-" + i + " : ", metrics.accuracy_score(y_test, y_pred))
#     print("Precision : ", metrics.precision_score(y_test, y_pred, average='micro'))
#     print("Recall : ", metrics.recall_score(y_test, y_pred, average='micro'))