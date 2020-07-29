import nltk
import inflect
import contractions
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder


# First function is used to denoise text
def denoise_text(text):
    # Strip html if any. For ex. removing <html>, <p> tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Replace contractions in the text. For ex. didn't -> did not
    text = contractions.fix(text)
    return text

def clean_tweet(text):
  ''' 
  Utility function to clean tweet text by removing links, special characters 
  using simple regex statements. 
  '''
  return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())


def remove_extra_spaces(text):
  text = text.replace("\n", "").replace("\t", "").replace('\\"',"") #.replace('') #.replace('"', "")
  text = text.replace(',', "").replace('"', "")
  text = re.sub(' +', ' ', text)
  return text


def normalize(text):
  text = remove_extra_spaces(text)
  text = denoise_text(text)
  text = clean_tweet(text)
  return text

def get_text(j_data):
  text = ""
  for sentence in j_data['summary']['sentences']:
    text = text + ". "+ sentence
    text = normalize(text)


  return text

def get_polarity(j_data):
  polarity = j_data['sentiment']['body']['polarity']
  if polarity == 'positive':
      polarity = 1 
  else:
     polarity = 0 

  return polarity
