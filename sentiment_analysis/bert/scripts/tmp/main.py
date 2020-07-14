# =========================== Import Libraries ================================================

import os
from mylib.utils import load_datasets
from mylib.sentiment_detection_data import SentimentDetectionData
from mylib.sentiment_detection_data import create_model

import tensorflow as tf
from tensorflow import keras
from bert.tokenization.bert_tokenization import FullTokenizer
import datetime
# =========================== Import Libraries ================================================


# =========================== Defining Constants ==============================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_base_dir = "/home/shravan/Downloads/train/"
valid_base_dir = "/home/shravan/Downloads/valid/"

bert_abs_path = '/home/shravan/Downloads/'
bert_model_name = 'multi_cased_L-12_H-768_A-12'

bert_ckpt_dir = os.path.join(bert_abs_path, bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, 'bert_model.ckpt')
bert_config_file = os.path.join(bert_ckpt_dir, 'bert_config.json')
# =========================== Defining Constants =============================================

# =========================== Data Preparation ===============================================
train, test = load_datasets(train_base_dir, valid_base_dir)

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))
classes = train.polarity.unique().tolist()

data = SentimentDetectionData(train, test, tokenizer, classes, max_seq_len=128)
# =========================== Data Preparation ===============================================

# =========================== Building and Training Model ====================================
model = create_model(data.max_seq_len, bert_ckpt_file, bert_config_file, classes)
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')]
)


log_dir = 'log/sentiment_detection' + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
    x=data.train_x,
    y=data.train_y,
    validation_split=0.1,
    batch_size=16,
    shuffle=True,
    epochs=10,
)
# =========================== Building and Training Model ====================================

# =========================== Saving trained Model ===========================================
check_point_path = '/home/shravan/dissertation/bert_model/saved_model/1'
tf.saved_model.save(model, check_point_path)
# =========================== Saving trained Model ===========================================

