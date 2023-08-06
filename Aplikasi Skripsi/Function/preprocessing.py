import pandas as pd 
import numpy
import re
import json
import joblib
import os
import lib.tesaurus.tesaurus as ts
import matplotlib.pyplot as plt
import matplotlib
import statistics as st
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

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

def slangWordFilter(list, list_slangwords):
    filtered_list = []
    
    for word in list:
        if word in list_slangwords.keys():
            filtered_list.append(list_slangwords[word])
        else:
            filtered_list.append(word)
    
    return filtered_list

def prepareData(data_source_url):   
    if not os.path.exists('Process'):
        os.makedirs('Process')

    read_file = pd.read_excel(data_source_url)
    read_file.to_csv("Process/Data.csv", index=1, header=True)
    read_file = read_file.drop('Sentiment', axis=1)
    read_file.to_excel("Process/DataTest.xlsx", index=1, header=True)
    
    tweets = pd.read_csv("Process/Data.csv")

    features = tweets.loc[:, 'Tweet'].values
    labels = tweets.loc[:, 'Sentiment'].values

    return features, labels
   
def preprocessing(features):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    list_stopwords = stopwords.words('indonesian')

    # new_stopwords = open('../../Function/lib/NLP_bahasa_resources/combined_stop_words.txt').read().split("\n")
    # list_stopwords.extend(new_stopwords)

    list_slangwords = json.load(open('../../Function/lib/NLP_bahasa_resources/combined_slang_words.txt'))
    
    processed_features = []

    for sentences in features:

        # Remove Mention, links and Hastag
        temporary = re.sub('http://\S+|https://\S+|@\S+|#\S+|##\S+', ' ', str(sentences))

        # Remove Special Character
        temporary = re.sub(r'\W', ' ', str(temporary))
        
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

        # Tokenize
        token_result = word_tokenize(temporary)
        
        # Change Slang Words
        slangwords_result = slangWordFilter(token_result, list_slangwords)
        
        # Remove Stop Words
        stopwords_result = [word for word in slangwords_result if not word in list_stopwords]
        
        # Synonym Words
        synonym = synonymWordFilter(stopwords_result)
        
        # Back To String
        temporary = ' '.join(map(str, synonym))
        
        # Stemming Bahasa Indonesia
        temporary = stemmer.stem(temporary)
        
        # Remove Multiple Word
        # temporary = ' '.join(set(temporary.split()))
            
        # Tambahkan kedalam list
        processed_features.append(temporary)
  
    return processed_features

def generateWordCloud(text, name, mode):
    list_stopwords = stopwords.words('indonesian')
    new_stopwords = open('../../Function/lib/NLP_bahasa_resources/combined_stop_words.txt').read().split("\n")
    list_stopwords.extend(new_stopwords)

    wordcloud = WordCloud(max_words=100, height=400, width=800, background_color="black").generate(text)
    matplotlib.use('agg')            
    plt.figure(figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud, interpolation="nearest")
    plt.axis("off")
    plt.savefig(f"{mode}Data/{name + mode}WordCloud.png")
    matplotlib.pyplot.close()

def trainModel(labels, processed_features):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = svm.SVC(kernel="rbf")
    vectorizers = TfidfVectorizer()
    tfIdf_svm = Pipeline([('tfidf', vectorizers), ('svc', clf)])
    processed_features = numpy.array(processed_features)
    all_tfidf = vectorizers.fit_transform(processed_features)
    
    positiveWords = ''
    netralWords = ''
    negativeWords = ''
    best_fold = {
        "f1": 0
    }
    all_fold = []
    temp_fold = {}
        
    i = 1
    for train_index, test_index in kf.split(processed_features, labels):
        X_train, X_test = processed_features[train_index], processed_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        tfIdf_svm.fit(X_train, y_train)
        y_predict = tfIdf_svm.predict(X_test)
        X_test = tfIdf_svm.named_steps['tfidf'].transform(X_test)
        
        rows, cols = X_test.nonzero()
        data = {"row": rows, "col": cols, "data": X_test.data}
        df = pd.DataFrame(data=data)
        mean_list = []
        for j in range(0, len(numpy.unique(rows))):
            temp_list = df.loc[df['row'] == j, 'data'].values.tolist()
            mean_list.append(sum(temp_list))

        X_test_numpy = numpy.array(mean_list)

        # X_train = vectorizers.fit_transform(X_train)
        # X_test = vectorizers.transform(X_test)
        
        # Train
        # clf.fit(X_train, y_train)
        # y_predict = clf.predict(X_test)
                        
        # Prepare Data
        # pca = PCA(n_components=2)
        # X_test_np = numpy.array(X_test.todense())
        # timesTen = lambda x: x * 10
        # X_test_np = timesTen(X_test_np)
        
        # pca.fit(X_test_np, y_test)
        # data2D = pca.fit_transform(X_test_np, y_test)
        # clf.fit(data2D, y_test)
        
        # x_min, x_max = data2D[:, 0].min() - 1, data2D[:, 0].max() + 1
        # y_min, y_max = data2D[:, 1].min() - 1, data2D[:, 1].max() + 1
        # xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.02), numpy.arange(y_min, y_max, 0.02))
                
        # Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)
        
        # plt.subplot(3, 4, i)
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)
        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
        # plt.scatter(data2D[y_test == 1, 0], data2D[y_test == 1, 1], c=y_test[y_test == 1], edgecolor="k")
        # plt.scatter(data2D[y_test == 0, 0], data2D[y_test == 0, 1], c=y_test[y_test == -0], edgecolor="k")
        # plt.scatter(data2D[y_test == -1, 0], data2D[y_test == -1, 1], c=y_test[y_test == -1], edgecolor="k")
        # plt.axis([x_min, x_max, y_min, y_max])
        # plt.savefig('TrainData/TrainChart.png')
        # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        # markers = ('s', 'x', 'o', '^', 'v')
        # for idx, cl in enumerate(numpy.unique(y_test)):
        #     plt.scatter(x = data2D[y_test == cl, 0], y = data2D[y_test == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl)
        #     plt.scatter(data2D[:, 0], data2D[:, 1], c=labels[test_index], marker='x', cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        #     plt.savefig('TrainData/TrainChart.png')
                  
        countNetral = 0  
        countPositive = 0  
        countNegative = 0  
  
        falsePositive = 0  
        falseNegative = 0  
        falseNetral = 0  
        truePositive = 0  
        trueNegative = 0  
        trueNetral = 0
            
        for index, content in enumerate(y_predict):
            # Prediksi Salah
            # if content != labels[test_index][index]:        
            #     print(f"Fold {i} Isi {processed_features[test_index[index]]} Labels {content} Correct {labels[test_index[index]]}")
            
            # Jumlah Positif, Negatif dan Netral
            if content == 0:
                countNetral += 1  
                netralWord = processed_features[test_index[index]]
                netralWords += ' ' + netralWord
            elif content == 1:
                countPositive += 1            
                positiveWord = processed_features[test_index[index]]
                positiveWords += ' ' + positiveWord
            elif content == -1:
                countNegative += 1
                negativeWord = processed_features[test_index[index]]
                negativeWords += ' ' + negativeWord

            # Jumlah True Positive, False Positive, True Netral, False Netral, True Negative, False Negative
            if content == 0:
                if content == labels[test_index[index]]:
                    trueNetral += 1
                else:
                    falseNetral += 1
            elif content == 1:
                if content == labels[test_index[index]]:
                    truePositive += 1
                else:
                    falsePositive += 1
            elif content == -1:
                if content == labels[test_index[index]]:
                    trueNegative += 1
                else:
                    falseNegative += 1

        # Display Confusion Matrix
        confusion_matrix = metrics.confusion_matrix(y_test, y_predict)
        confusion_matrix = numpy.flipud(confusion_matrix)
        confusion_matrix = numpy.fliplr(confusion_matrix)
        score_cm = metrics.classification_report(y_test, y_predict, zero_division=0, output_dict=True)
        accuracy = metrics.accuracy_score(y_test, y_predict)

        if best_fold['f1'] < score_cm['macro avg']['f1-score']:
            best_fold['fold'] = i
            best_fold['clf'] = tfIdf_svm
            best_fold['accuracy'] = accuracy
            best_fold['precision'] = score_cm['macro avg']['precision']
            best_fold['recall'] = score_cm['macro avg']['recall']
            best_fold['f1'] = score_cm['macro avg']['f1-score']
            best_fold['score_cm'] = score_cm
            best_fold['confusion_matrix'] = confusion_matrix.tolist()
            best_fold['count'] = [countPositive, countNetral, countNegative]
            best_fold['true'] = [truePositive, trueNetral, trueNegative]
            best_fold['false'] = [falsePositive, falseNetral, falseNegative]
            best_fold['x_test'] = X_test_numpy
            best_fold['y_test'] = y_test
            
            matplotlib.use('agg')
            result = numpy.column_stack((best_fold['x_test'].data, best_fold['y_test']))
            model = clf.fit(result[:, :2], best_fold['y_test'])
            display = DecisionBoundaryDisplay.from_estimator(model, result[:, :2], response_method="predict", alpha=0.5)
            display.plot(plot_method="contourf", xlabel="Test Features", ylabel="Predicted Labels")
            display.ax_.scatter(result[best_fold['y_test'] == -1, 0], result[best_fold['y_test'] == -1, 1], edgecolors="black", marker='X')
            display.ax_.scatter(result[best_fold['y_test'] == 0, 0], result[best_fold['y_test'] == 0, 1], edgecolors="black", marker='o')
            display.ax_.scatter(result[best_fold['y_test'] == 1, 0], result[best_fold['y_test'] == 1, 1], marker='+')
            plt.savefig('TrainData/TrainChart.png')
            
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Positive", "Netral", "Negative"])
            cm_display.plot()
            plt.savefig('TrainData/TrainPlot.png')
            matplotlib.pyplot.close()
        
        temp_fold['fold'] = i
        temp_fold['accuracy'] = accuracy
        temp_fold['precision'] = score_cm['macro avg']['precision']
        temp_fold['recall'] = score_cm['macro avg']['recall']
        temp_fold['f1'] = score_cm['macro avg']['f1-score']
        temp_fold['score_cm'] = score_cm
        temp_fold['confusion_matrix'] = confusion_matrix.tolist()
        temp_fold['count'] = [countPositive, countNetral, countNegative]
        temp_fold['true'] = [truePositive, trueNetral, trueNegative]
        temp_fold['false'] = [falsePositive, falseNetral, falseNegative]
        
        all_fold.append(temp_fold.copy())

        i += 1
        
    if not os.path.exists('Model'):
        os.mkdir('Model')
        
    joblib.dump(best_fold['clf'], 'Model/svm.pkl')

    # sorted_fold = sorted(all_fold, key=lambda x: x['f1'], reverse=True)
    # for index, fold in enumerate(sorted_fold):
    #     print(f"Urutan ke {index + 1} Fold ke {fold['fold']} f1-score {fold['f1']}")
        
    df_value = []
    words_list = tfIdf_svm.named_steps['tfidf'].get_feature_names_out()
    for index, word in enumerate(words_list):
        df = numpy.sum(all_tfidf[:, index] > 0)
        df_value.append({'word': word, 'df': df})
    
    return best_fold, all_fold, positiveWords, netralWords, negativeWords, df_value

def startTrain(data_source_url):
    features, labels = prepareData(data_source_url=data_source_url)
    processed_features = preprocessing(features=features)
    best_fold, all_fold, positiveWords, netralWords, negativeWords, df_value = trainModel(labels=labels, processed_features=processed_features)
    generateWordCloud(positiveWords, "Positive", "Train")
    generateWordCloud(netralWords, "Netral", "Train")
    generateWordCloud(negativeWords, "Negative", "Train")

    return best_fold, all_fold, df_value


