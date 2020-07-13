import os
import math
import datetime

from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import seaborn as sns

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

train_base_dir = "/home/shravan/Downloads/train/"
valid_base_dir = "/home/shravan/Downloads/valid/"
train_count = 11


def load_datasets():
    train_df = pd.DataFrame()
    for name in os.listdir(train_base_dir):
        file_path = os.path.join(train_base_dir, name)
        train_df = pd.concat([train_df,
                              pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True
                             )

    valid_df = pd.DataFrame()
    for name in os.listdir(valid_base_dir):
        file_path = os.path.join(valid_base_dir, name)
        valid_df = pd.concat([valid_df,
                              pd.read_csv(file_path, sep=',', names=["sentences", "polarity"])],
                             ignore_index=True
                             )

    return train_df, valid_df


train, test = load_datasets()

bert_abs_path = '/home/shravan/Downloads/'
bert_model_name = 'multi_cased_L-12_H-768_A-12'

bert_ckpt_dir = os.path.join(bert_abs_path, bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


# Preprocessing
class IntentDetectionData:
    DATA_COLUMN = 'sentences'
    LABEL_COLUMN = 'polarity'

    def __init__(self, train, test, tokenizer: FullTokenizer, classes, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = 0
        self.classes = classes

        # print(train[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index())
        train, test = map(lambda df: df.reindex(df[IntentDetectionData.DATA_COLUMN].str.len().sort_values().index),
                          [train, test])

        ((self.train_x, self.train_y), (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []

        for _, row in tqdm(df.iterrows()):
            text, label = row[IntentDetectionData.DATA_COLUMN], row[IntentDetectionData.LABEL_COLUMN]
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))

        return np.array(x), np.array(y)

    def _pad(self, ids):
        x = []
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))

        return np.array(x)


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))

t = tokenizer.tokenize('ಶುಭ ದಿನ')
print(t)

ds = tokenizer.convert_tokens_to_ids(t)
print(ds)


def create_model(max_seq_len, bert_ckpt_file):
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        bert = BertModelLayer.from_params(bert_params, name='bert')

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='input_ids')
    print('----intput_ids', input_ids)
    bert_output = bert(input_ids)

    print('bert shape', bert_output.shape)

    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation='tanh')(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation='softmax')(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    return model


classes = train.polarity.unique().tolist()

data = IntentDetectionData(train, test, tokenizer, classes, max_seq_len=128)

print(data.train_x.shape)

# Training:

model = create_model(data.max_seq_len, bert_ckpt_file)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')]
)

log_dir = 'log/intent_detection' + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    x=data.train_x,
    y=data.train_y,
    validation_split=0.1,
    batch_size=16,
    shuffle=True,
    epochs=5,
)

check_point_path = '/home/shravan/dissertation/bert_model'
tf.saved_model.save(model, check_point_path)
# model.save(check_point_path)
