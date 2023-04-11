import joblib
import pandas as pd
import preprocessing

def loadModel(path):
    clf = joblib.load(path)
    
    return clf

def loadData(path):
    tweets = pd.read_excel(path, sheet_name="Sheet1")
    features = tweets.loc[:, 'Tweet'].values
    
    return features

def generateDataWordCloud(processed_features, predict_result):
    positiveWords = ''
    netralWords = ''
    negativeWords = ''
    countNetral = 0
    countPositive = 0
    countNegative = 0

    for index, content in enumerate(predict_result):
        if content == 0:
            countNetral += 1  
            netralWord = processed_features[index]
            netralWords += ' ' + netralWord
        elif content == 1:
            countPositive += 1            
            positiveWord = processed_features[index]
            positiveWords += ' ' + positiveWord
        elif content == -1:
            countNegative += 1
            negativeWord = processed_features[index]
            negativeWords += ' ' + negativeWord
    
    return positiveWords, netralWords, negativeWords, countPositive, countNetral, countNegative

def startPredict(pathModel, pathData):
    clf = loadModel(pathModel)
    features = loadData(pathData)
    processed_features = preprocessing.preprocessing(features=features)
    print(clf)
    result = clf.predict(processed_features)
    positiveWords, netralWords, negativeWords, countPositive, countNetral, countNegative = generateDataWordCloud(processed_features=processed_features, predict_result=result)

    if countPositive > 0:
        preprocessing.generateWordCloud(positiveWords, "Positive", "Test")
    if countNegative > 0:
        preprocessing.generateWordCloud(negativeWords, "Negative", "Test")
    if countNetral > 0:
        preprocessing.generateWordCloud(netralWords, "Netral", "Test")

    data = {'Predict Result': result, 'Tweet': features}

    df = pd.DataFrame(data)
    df.to_excel('Process/DataPredict.xlsx')

    return positiveWords, netralWords, negativeWords, countPositive, countNetral, countNegative