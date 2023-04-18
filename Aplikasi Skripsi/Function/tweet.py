import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import os
from zipfile import ZipFile

def query(name, start_date, end_date):
    query = f'"{name}" Presiden until:{end_date} since:{start_date}'
    
    return appendData(query)
    
def appendData(query):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query, maxEmptyPages=1000).get_items():
        tweets.append([tweet.date, ' ',str(tweet.rawContent)])

    return tweets

def insertToExcel(name, start_date, end_date):
    tweets = query(name, start_date, end_date)
    
    df = pd.DataFrame(tweets, columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    
    if not os.path.exists("Data"):
        os.makedirs("Data")
    
    if os.path.exists(f"Data/output_{name}.xlsx"):
        os.remove(f"Data/output_{name}.xlsx")
        
    df.to_excel(f"Data/output_{name}.xlsx")