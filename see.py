#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab


# base_dir = 'D:\\work\\Dail_NLP\\sentiment_analysis\\data\\'
# train_dir = os.path.join(base_dir, 'train_data1.csv')
# test_dir = os.path.join(base_dir, 'add_test1.csv')
# val_dir = os.path.join(base_dir, 'val_data1.csv')
# vocab_dir = os.path.join(base_dir, 'add_vocab1.txt')
base_dir = 'D:\\work\\Dail_NLP\\sentiment_analysis\\data\\'
train_dir = os.path.join(base_dir, 'dev_train.csv')
test_dir = os.path.join(base_dir, 'add_test.csv')
val_dir = os.path.join(base_dir, 'dev_val.csv')
vocab_dir = os.path.join(base_dir, 'add_vocab.txt')
model_dir = os.path.join(base_dir, 'vocab_model')

# config = TCNNConfig()
# categories, cat_to_id = read_category()
# print(cat_to_id)
# words, word_to_id = read_vocab(vocab_dir)
# config.vocab_size = len(words)
#
# x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
from gensim.models import word2vec
with open(vocab_dir, 'r', encoding='utf-8') as f:
    vocab = f.read().split('\n')
import tensorflow as tf
vocab_model = word2vec.Word2Vec(vocab, size=256, window=5, min_count=1, workers=4)
vocab_model.save(model_dir)
embedding = []
more_v = []
for w in vocab:
    more_v.append(w)
    embedding.append(vocab_model[w])

emb1 = np.array(embedding)
ha = tf.constant(value=emb1, dtype=tf.float32)


import numpy as np
import tensorflow as tf
from gensim.models import word2vec
vocab_model = word2vec.Word2Vec.load('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\vocab_model')
embedding0 = []
with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().split('\n')
for w in vocab:
    embedding0.append(vocab_model[w])
emb1 = np.array(embedding0)
embedding = tf.constant(value=emb1, dtype=tf.float32)

vocab_model = word2vec.Word2Vec.load('D:\\work\\Dail_NLP\\sentiment_analysis\\model\\sohu_single_word.model')
