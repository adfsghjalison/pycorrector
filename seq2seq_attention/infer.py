# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""
import sys

sys.path.append('../..')
import os

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import load_word_dict
from pycorrector.seq2seq_attention.evaluate import gen_target
from pycorrector.seq2seq_attention.seq2seq_attn_model import Seq2seqAttnModel


class Inference(object):
    def __init__(self, save_vocab_path='', attn_model_path='', maxlen=400):
        if os.path.exists(save_vocab_path):
            self.char2id = load_word_dict(save_vocab_path)
            self.id2char = {int(j): i for i, j in self.char2id.items()}
            self.chars = set([i for i in self.char2id.keys()])
        else:
            print('not exist vocab path')
        seq2seq_attn_model = Seq2seqAttnModel(self.chars, attn_model_path=attn_model_path, hidden_dim=config.rnn_hidden_dim, use_gpu=config.use_gpu)
        self.model = seq2seq_attn_model.build_model()
        self.maxlen = maxlen

    def infer(self, sentence):
        return gen_target(sentence, self.model, self.char2id, self.id2char, self.maxlen, topk=3)


if __name__ == "__main__":

    cf = tf.ConfigProto()
    cf.gpu_options.allow_growth=True
    session = tf.Session(config=cf)
    KTF.set_session(session)

    inference = Inference(save_vocab_path=config.save_vocab_path,
                          attn_model_path=config.attn_model_path,
                          maxlen=400)

    if config.infer_path != None:
        f = open(config.output_path, 'w')
        for i, l in enumerate(open(config.infer_path)):
            if i < config.infer_count:
                continue
            l = l.strip()
            target = inference.infer(l)
            f.write(l+'\n'+target+'\n\n')
            if i % 100 == 0:
                print(i)
    else:
        print('start infering ...')
        sent = input('').strip()
        while sent != 'q':
            print(inference.infer(sent))
            sent = input('').strip()

