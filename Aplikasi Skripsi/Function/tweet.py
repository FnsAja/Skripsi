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
        print(tweet.date)
        tweets.append([' ', re.sub('http://\S+|https://\S+|@\S+|#\S+|##\S+', '', str(tweet.rawContent))])

    return tweets

def insertToExcel(name, start_date, end_date):
    tweets = query(name, start_date, end_date)
    
    df = pd.DataFrame(tweets, columns=['Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    df1 = df.drop('Sentiment', axis=1)
    
    if not os.path.exists("Data"):
        os.makedirs("Data")
    
    if os.path.exists(f"Data/output_{name}.xlsx"):
        os.remove(f"Data/output_{name}.xlsx")

    if os.path.exists(f"Data/output_{name}_DataTest.xlsx"):
        os.remove(f"Data/output_{name}_DataTest.xlsx")
        
    df.to_excel(f"Data/output_{name}.xlsx")
    df1.to_excel(f"Data/output_{name}_DataTest.xlsx")

    with ZipFile(f'Data/Data_{name}.zip', 'w') as zipFile:
        zipFile.write(f'Data/output_{name}.xlsx')
        zipFile.write(f'Data/output_{name}_DataTest.xlsx')