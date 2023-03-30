import joblib
import os
import pandas as pd
import preprocessing

def loadModel(path):
    clf = joblib.load(path)
    
    return clf

def loadData(path):
    tweets = pd.read_csv(path)
    features = tweets.loc[:, 'Tweet'].values
    
    return features

def startPredict(pathModel, pathData):
    clf = loadModel(pathModel)
    features = loadData(pathData)
    processed_features = preprocessing.preprocessing(features=features)
    calculated_features = preprocessing.weighting(processed_features)
    result = clf.predict(calculated_features)

    if not os.path.exists('TestData'):
        os.mkdir('TestData')

    data = {'Predict Result': result, 'Tweet': features}

    df = pd.DataFrame(data)
    df.to_excel('Process/DataPredict.xlsx')

    return df