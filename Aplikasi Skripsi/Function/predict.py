import joblib
import pandas as pd
import preprocessing

clf = joblib.load("svm.pkl")
tweets = pd.read_csv("DataTest.csv", sep="delimiter", header=0, engine="python")
features = tweets.iloc[:, 0].values

processed_features = preprocessing.preprocessing(features=features)
calculated_features = preprocessing.weighting(processed_features)

print(clf.predict(calculated_features))