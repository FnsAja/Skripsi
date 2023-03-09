import snscrape.modules.twitter as sntwitter
import pandas as pd
import re
import os

def query(name):
    jan = f'"{name}" Presiden until:2023-01-31 since:2023-01-01'
    feb = f'"{name}" Presiden until:2023-02-28 since:2023-02-01'
    mar = f'"{name}" Presiden until:2023-03-31 since:2023-03-01'
    
    tweets = [appendData(jan), appendData(feb), appendData(mar)]
    return tweets
    
def appendData(query):
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query, maxEmptyPages=1000).get_items():
        print(tweet.date)
        tweets.append([str(tweet.date).split(' ')[0], ' ', re.sub('http://\S+|https://\S+|@\S+|#\S+|##\S+', '', str(tweet.rawContent))])

    return tweets

def insertToExcel(tweets, name):
    df1 = pd.DataFrame(tweets[0], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    df2 = pd.DataFrame(tweets[1], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    df3 = pd.DataFrame(tweets[2], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    if os.path.exists(f"output_{name}.xlsx"):
        os.remove(f"output_{name}.xlsx")
        
    df1.to_excel(f"output_{name}.xlsx_Jan")
    df2.to_excel(f"output_{name}.xlsx_Feb")
    df3.to_excel(f"output_{name}.xlsx_Mar")
        
    print("success")

def insertToCSV(tweets, name):
    df1 = pd.DataFrame(tweets[0], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    df2 = pd.DataFrame(tweets[1], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    df3 = pd.DataFrame(tweets[2], columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
    if os.path.exists(f"output_{name}.csv"):
        os.remove(f"output_{name}.csv")
        
    df1.to_csv(f"output_{name}.csv_Jan")
    df2.to_csv(f"output_{name}.csv_Feb")
    df3.to_csv(f"output_{name}.csv_Mar")
        
    print("success")

# name = "Prabowo Subianto"
# tweets = query(name)
# insertToExcel(tweets, name)
# insertToCSV(tweets, name)

# name = "Anies Baswedan"
# tweets = query(name)
# insertToExcel(tweets, name)
# insertToCSV(tweets, name)

# name = "Prabowo Subianto"
# tweets = query(name)
# insertToExcel(tweets, name)
# insertToCSV(tweets, name)