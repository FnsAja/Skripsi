import tweepy

api_key='E0fGDpfVvcNHtwnrT39AUOcUw'
api_key_secret='CSwxBJYx1SPVg8rkbGHj5PP5iIBFflsfwvjD2t4appk3Fk2AfC' 
access_token='1367654371261964291-SHqjCGhyETxy7iVJxBLlC3JQ79m7PB' 
access_token_secret='fZt3owSuH5ROpSfCslfF0pxlg6f6MZQecgfX1zwabFzWK'
bearer_token='AAAAAAAAAAAAAAAAAAAAAGvpmwEAAAAAnC9fMo3yS925pX8f%2BE7jC2UYXkc%3D2zBlsBCLFrZjagE31DgQrnLQup1dqPcrHSeDkIYGefWaf3ixFW'                     

authenticator = tweepy.OAuthHandler(api_key, api_key_secret)
authenticator.set_access_token(access_token, access_token_secret)

api = tweepy.API(authenticator, wait_on_rate_limit=True)

tweets = api.search_tweets('Presiden')
print(tweets)

for tweet in tweets:
    print(tweet)