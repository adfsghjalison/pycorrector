# -*- coding: utf-8 -*-
"""
@author:XuMing（xuming624@qq.com)
@description: 
"""

import numpy as np
from keras.callbacks import Callback, EarlyStopping

from pycorrector.seq2seq_attention import config
from pycorrector.seq2seq_attention.corpus_reader import str2id, id2str
from pycorrector.seq2seq_attention.reader import GO_TOKEN, EOS_TOKEN


def gen_target(input_text, model, char2id, id2char, maxlen=400, topk=3, max_target_len=50):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    xid = np.array([str2id(input_text, char2id, maxlen)] * topk)  # 输入转id
    yid = np.array([[char2id[GO_TOKEN]]] * topk)  # 解码均以GO开始
    scores = [0] * topk  # 候选答案分数
    for i in range(max_target_len):  # 强制要求target不超过maxlen字
        proba = model.predict([xid, yid])[:, i, :]  # 预测
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        arg_topk = log_proba.argsort(axis=1)[:, -topk:]  # 每一项选出topk
        _yid = []  # 暂存的候选目标序列
        _scores = []  # 暂存的候选目标序列得分
        if i == 0:
            for j in range(topk):
                _yid.append(list(yid[j]) + [arg_topk[0][j]])
                _scores.append(scores[j] + log_proba[0][arg_topk[0][j]])
        else:
            for j in range(len(xid)):
                for k in range(topk):  # 遍历topk*topk的组合
                    _yid.append(list(yid[j]) + [arg_topk[j][k]])
                    _scores.append(scores[j] + log_proba[j][arg_topk[j][k]])
            _arg_topk = np.argsort(_scores)[-topk:]  # 从中选出新的topk
            _yid = [_yid[k] for k in _arg_topk]
            _scores = [_scores[k] for k in _arg_topk]
        yid = []
        scores = []
        for k in range(len(xid)):
            if _yid[k][-1] == char2id[EOS_TOKEN]:  # 找到<end>就返回
                return id2str(_yid[k][1:-1], id2char)
            else:
                yid.append(_yid[k])
                scores.append(_scores[k])
        yid = np.array(yid)
    # 如果maxlen字都找不到EOS，直接返回
    return id2str(yid[np.argmax(scores)][1:-1], id2char)


class Evaluate(Callback):
    def __init__(self, model, attn_model_path, char2id, id2char, maxlen, log_file, trained):
        super(Evaluate, self).__init__()
        self.loss = 0.0
        self.lowest = 1e10
        self.trained = trained
        self.model = model
        self.attn_model_path = attn_model_path
        self.char2id = char2id
        self.id2char = id2char
        self.maxlen = maxlen
        self.log_file = log_file

    def on_batch_end(self, batch, logs=None):
        self.loss += logs['loss']
        if batch % config.print_step == 0:
            print('loss: {}'.format(self.loss / config.print_step))
            sents = ['他們知不道吸菸對未成年年的影響會造成的各種害處',
                    '如果它呈出不太香的顏色',
                    '可是我覺得飢餓是用科學機技來能解決問題']
            print('')
            for sent in sents:
                target = gen_target(sent, self.model, self.char2id, self.id2char, self.maxlen)
                print(sent)
                print(target)
            print('')
            """
            # 保存最优结果
            if batch % config.save_step == 0:
                if logs['loss'] <= self.lowest:
                    self.lowest = logs['loss']
                    self.model.save_weights(self.attn_model_path)
                    print('Saving model ...')
                else:
                    print('Not Saving ...')
            """
            self.loss = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优结果
        ep = self.trained + epoch + 1
        print('Saving model ...')
        self.model.save_weights(self.attn_model_path)
        open(self.log_file, 'w').write(str(ep))
        if ep % 20 == 0:
            self.model.save_weights("{}.{}".format(self.attn_model_path, ep))
        """
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
        else:
            print('Not Saving ...')
        """

