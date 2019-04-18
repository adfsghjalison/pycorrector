# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: Use CGED corpus
import os, sys
from argparse import ArgumentParser

# CGED chinese corpus
raw_train_paths = [
    '../data/cn/CGED/CGED18_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED17_HSK_TrainingSet.xml',
    '../data/cn/CGED/CGED16_HSK_TrainingSet.xml',
    #'../data/cn/CGED/sample_HSK_TrainingSet.xml',
]
data_dir = 'data'
model_dir = 'model'
output_dir = 'output'
train_path = os.path.join(data_dir, 'train')
#train_path = os.path.join(data_dir, 'temp')
test_path = os.path.join(data_dir, 'test')
infer_path1 = os.path.join(data_dir, 'infer')
infer_path2 = os.path.join(data_dir, 'infer_all')
infer_count = 0

parser = ArgumentParser()
parser.add_argument("-load", dest='load', default=-1)
parser.add_argument("-infer", dest='infer', default=None)
args = parser.parse_args()

infer_path = args.infer
if infer_path != None:
    infer_path = infer_path1 if infer_path == 1 else infer_path2

output_path = os.path.join(output_dir, 'output' if infer_path==infer_path1 else 'output_all')

# seq2seq_attn_train config
save_vocab_path = os.path.join(data_dir, 'vocab')
attn_model_path = os.path.join(model_dir, 'attn_model')
log = os.path.join(model_dir, 'log')
if args.load != -1:
    attn_model_path += ('.'+args.load)
    output_path += ('.'+args.load)

# config
batch_size = 32
print_step = 1000
save_step = 10000
epochs = 100
rnn_hidden_dim = 256
minlen = 2
maxlen = 40
dropout = 0.2
use_gpu = True

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

