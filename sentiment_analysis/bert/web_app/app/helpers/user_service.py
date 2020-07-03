import os
import math
import datetime
import json

import numpy as np
import tensorflow as tf


import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer
import requests



def get_tokenizer():
  # ================ Load Tokenizer =============================================
  bert_abs_path = '/home/shravan/Downloads/'
  bert_model_name = 'multi_cased_L-12_H-768_A-12'

  bert_ckpt_dir = os.path.join(bert_abs_path, bert_model_name)
  bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
  bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


  tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))
  # ================ Load Tokenizer =============================================
  return tokenizer


def convert_text_to_tokens(tokenizer, text):
  # ================ Convert Text to Tokens ===================================== 
  sentences = [text]
  classes = ['Negative', 'Positive']
  pred_tokens = map(tokenizer.tokenize, sentences)
  pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
  pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

  pred_token_ids = map(lambda tids: tids + [0] * (128 - len(tids)), pred_token_ids)
  pred_token_ids = list(pred_token_ids)
  tokens = pred_token_ids
  # ================ Convert Text to Tokens ===================================== 
  return tokens


def predict_sentiment(tokens):
  # ================ Predict Sentiment from Tokens ==============================
  URL ='http://localhost:8501/v1/models/saved_model:predict'
  headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
  body = {'instances': tokens}

  print("----->body--->", body)
  response = requests.post(URL, json=body, headers=headers)
  probability = response.json()['predictions']
  index = np.argmax(probability)
  # ================ Predict Sentiment from Tokens ==============================
  if index == 1:
    return 'Positive'
  return 'Negative'

def infer(review):
  tokenizer = get_tokenizer() 
  tokens = convert_text_to_tokens(tokenizer, review)
  sentiment = predict_sentiment(tokens)

  return sentiment


def analyse(tweets):
  tokenizer = get_tokenizer()
  data = []
  for tweet in tweets:
    tokens = convert_text_to_tokens(tokenizer, tweet)
    sentiment = predict_sentiment(tokens)
    data.append([tweet, sentiment])

  return data
