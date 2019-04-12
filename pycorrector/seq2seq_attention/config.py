# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os

# CGED chinese corpus
raw_train_paths = [
    '../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    #'../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
data_dir = 'data'
model_dir = 'model'
# Training data path.
#train_path = os.path.join(data_dir, 'train.txt')
train_path = os.path.join(data_dir, 'try.txt')
# Validation data path.
test_path = os.path.join(data_dir, 'test')

# seq2seq_attn_train config
save_vocab_path = os.path.join(data_dir, 'vocab.txt')
attn_model_path = os.path.join(model_dir, 'attn_model.weight')

# config
batch_size = 64
epochs = 1
rnn_hidden_dim = 256
maxlen = 400
dropout = 0.2
use_gpu = True

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

