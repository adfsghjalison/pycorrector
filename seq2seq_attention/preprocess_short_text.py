# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:
import sys

sys.path.append('../..')
from codecs import open
from xml.dom import minidom

from sklearn.model_selection import train_test_split

from pycorrector.seq2seq_attention import config
from pycorrector.tokenizer import segment

split_symbol = ['，', '。', '？', '！']


def split_2_short_text(sentence):
    for i in split_symbol:
        sentence = sentence.replace(i, '\t')
    return sentence.split('\t')


def parse_xml_file(path):
    print('Parse data from %s' % path)
    data_list = []
    dom_tree = minidom.parse(path)
    docs = dom_tree.documentElement.getElementsByTagName('DOC')
    for doc in docs:
        # Input the text
        text = doc.getElementsByTagName('TEXT')[0]. \
            childNodes[0].data.strip()
        # Input the correct text
        correction = doc.getElementsByTagName('CORRECTION')[0]. \
            childNodes[0].data.strip()

        texts = split_2_short_text(text)
        corrections = split_2_short_text(correction)
        if len(texts) != len(corrections):
            # print('error:' + text + '\t' + correction)
            continue
        for i in range(len(texts)):
            text_l = len(texts[i])
            corr_l = len(corrections[i])
            if text_l > config.maxlen or text_l < config.minlen \
                or corr_l > config.maxlen or corr_l < config.minlen:
                continue
            pair = [texts[i], corrections[i]]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, data_path, target=True):
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            #f.write(' '.join(src)+" +++$+++ "+' '.join(dst))
            f.write(src + '\n')
            if target:
                f.write(dst + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def transform_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1, shuffle=False)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)
    _save_data(test_lst, config.infer_path1, target=False)

if __name__ == '__main__':
    # train data
    data_list = []
    for path in config.raw_train_paths:
        data_list.extend(parse_xml_file(path))
    transform_corpus_data(data_list, config.train_path, config.test_path)
    _save_data(data_list, config.infer_path2, target=False)

