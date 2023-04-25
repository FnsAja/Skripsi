import tweepy
import pandas as pd

api_key='yKl3hYmYfaHQBJvQCuqdCf9p5'
api_key_secret='nPNrwA1JOktcttG5PIcKQMkHvJi8wsW8mcMwvq3kPyB7aEV0ar' 
access_token='1367654371261964291-S3z6yXDIpfcbhVJgTPvLinC97y5z8N' 
access_token_secret='mtkaEEaxUuO2df0J8DnetF97Q6gNhC4TJ9Rr2aXr4w28i'
bearer_token='AAAAAAAAAAAAAAAAAAAAAGvpmwEAAAAAdQrDrCZJP1s%2B8ZXhZDiHeOGzfwc%3D8JlPtwqCELhRjme6EPIcw04Cie0xOdO0CVnPKYMoUsOGHWmSGr'                     

#Pass in our twitter API authentication key
auth = tweepy.OAuth1UserHandler(
    api_key, api_key_secret,
    access_token, access_token_secret
)

#Instantiate the tweepy API
api = tweepy.API(auth, wait_on_rate_limit=True)

username = "john"
no_of_tweets =100


try:
    #The number of tweets we want to retrieved from the user
    tweets = api.user_timeline(screen_name=username, count=no_of_tweets)
    
    #Pulling Some attributes from the tweet
    attributes_container = [[tweet.created_at, tweet.favorite_count, tweet.source,  tweet.text] for tweet in tweets]

    #Creation of column list to rename the columns in the dataframe
    columns = ["Date Created", "Number of Likes", "Source of Tweet", "Tweet"]
    
    #Creation of Dataframe
    tweets_df = pd.DataFrame(attributes_container, columns=columns)

    print(tweets_df.head())
except BaseException as e:
    print('Status Failed On,',str(e))

# authenticator = tweepy.OAuthHandler(api_key, api_key_secret)
# authenticator.set_access_token(access_token, access_token_secret)

# api = tweepy.API(authenticator, wait_on_rate_limit=True)

# tweets = api.search_tweets('Presiden')
# print(tweets)

# for tweet in tweets:
#     print(tweet)

# client = tweepy.Client(bearer_token=bearer_token, consumer_key=api_key, consumer_secret=api_key_secret, access_token=access_token, access_token_secret=access_token_secret, wait_on_rate_limit=True)
client = tweepy.Client(bearer_token=bearer_token)
tweets = client.search_recent_tweets('Presiden')

# query = '#elonmusk -is:retweet lang:en'
# tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
# Get tweets that contain the hashtag #TypeKeywordHere
# -is:retweet means I don't want retweets
# lang:en is asking for the tweets to be in english
# print pulled tweets
for tweet in tweets.data:
    print('\n**Tweet Text**\n',tweet.text)
