import os
from bert.tokenization.bert_tokenization import FullTokenizer


bert_abs_path = '/home/shravan/Downloads/'
bert_model_name = 'multi_cased_L-12_H-768_A-12'

bert_ckpt_dir = os.path.join(bert_abs_path, bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))
t = tokenizer.tokenize('good ದಿನ')
print(t)
