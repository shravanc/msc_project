import tweepy
import csv
import pandas as pd
import re
import preprocessor as p
####input your credentials here
consumer_key = '0NfwllYwUfHydecGf6pHIfIa9'
consumer_secret = 'LVZUqFBT9XOxqRsTHVlcqFz1UtzXIN3a2SYOuCdegjN9OYifnV'
access_token = '86742546-zfcU9tgKracUz5j0zQH3gW4wht4AgMmt1fbgV9RSl'
access_token_secret = 'Eb60sdlg7dPGsW0QZJ0eENSMUNAOoSIoyZWRRWKMAkhqf'


def clean_tweet(tweet):
  tweet = re.sub(r'http\S+', '', tweet)
  tweet = re.sub(r'@\S+', '', tweet)
  tweet = re.sub(r'#\S+', '', tweet)
  return tweet


def get_tweets():
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)
  api = tweepy.API(auth,wait_on_rate_limit=True)
  #####United Airlines
  # Open/Create a file to append data

  data = []
  for tweet in tweepy.Cursor(api.search,q="#ಕೋವಿಡ್_19",
                           lang="kn",
                           since="2017-04-03").items(50):
    if not tweet.retweeted and ('RT @' not in tweet.text):
      text = clean_tweet(tweet.text)
      data.append(text)

  return data


def fetch():
  return get_tweets()



