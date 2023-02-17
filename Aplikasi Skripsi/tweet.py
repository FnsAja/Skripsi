import snscrape.modules.twitter as sntwitter
import pandas as pd
import re

query = '"Ganjar Pranowo" until:2023-03-31 since:2023-01-01'
tweets = []

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    print(tweet.date)
    tweets.append([str(tweet.date).split(' ')[0], ' ', re.sub('http://\S+|https://\S+|#\S+|##\S+|@\S+', '', str(tweet.rawContent))])

df = pd.DataFrame(tweets, columns=['Date', 'Sentiment', 'Tweet']).drop_duplicates(subset='Tweet')
df.to_excel("output_ganjar.xlsx")

print("success")