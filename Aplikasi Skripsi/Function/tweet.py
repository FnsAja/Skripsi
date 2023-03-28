import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import os

def query(name, start_date, end_date):
    query = f'"{name}" Presiden until:{end_date} since:{start_date}'
    
    return appendData(query)
    
def appendData(query):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query, maxEmptyPages=1000).get_items():
        print(tweet.date)
        tweets.append([str(tweet.date).split(' ')[0], ' ', re.sub('http://\S+|https://\S+|@\S+|#\S+|##\S+', '', str(tweet.rawContent))])

    return tweets

def insertToExcel(name, start_date, end_date):
    tweets = query(name, start_date, end_date)
    
    df = pd.DataFrame(tweets, columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    if os.path.exists(f"output_{name}.xlsx"):
        os.remove(f"output_{name}.xlsx")
        
    df.to_excel(f"output_{name}.xlsx")