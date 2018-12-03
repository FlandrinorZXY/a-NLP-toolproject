# coding: utf-8

import tensorflow as tf
from gensim.models import word2vec
import numpy as np
from tensorflow.contrib import layers
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import BasicLSTMCell

class ABLSTMConfig(object):
    """ABLSTM配置参数"""

    # embedding_dim = 64  # 词向量维度
    embedding_dim = 256  # 词向量维度
    # embedding_dim = 128  # 词向量维度


    # seq_length = 600  # 序列长度
    seq_length = 500  # 序列长度

    # num_classes = 10  # 类别数
    num_classes = 3  # 类别数

    num_filters = 256  # 卷积核数目
    # num_filters = 4  # 卷积核数目

    # kernel_size = 5  # 卷积核尺寸
    kernel_size = 5  # 卷积核尺寸


    # vocab_size = 5000  # 词汇表大小
    vocab_size = 40000  # 词汇表大小
    # vocab_size = 123768  # 词汇表大小
    # vocab_size = 36350


    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.8  # dropout保留比例
    # learning_rate = 1e-3  # 学习率
    learning_rate = 0.001  # 学习率

    # batch_size = 4  # 每批训练大小
    batch_size = 64  # 每批训练大小

    num_epochs = 300  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class ABLSTM(object):
    """文本分类，ABLSTM模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')#seq_length inputdata
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')#one_hot label
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.ablstm()
################################
    def AttentionLayer(self, inputs,name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.config.embedding_dim]), name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.config.embedding_dim, activation_fn=tf.nn.tanh)
            print(u_context.shape, h.shape)
            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.multiply(inputs, alpha)
            return atten_output
    # def cove_AttentionLayer(self, inputs,name):
    #     # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
    #     with tf.variable_scope(name):
    #         # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
    #         # 因为使用双向GRU，所以其长度为2×hidden_szie
    #         u_context = tf.Variable(tf.truncated_normal([self.config.seq_length-self.config.kernel_size+1]), name='u_context')
    #         # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
    #         h = layers.fully_connected(inputs, self.config.seq_length-self.config.kernel_size+1, activation_fn=tf.nn.tanh)
    #         print(u_context.shape, h.shape)
    #         # shape为[batch_size, max_time, 1]
    #         alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
    #         # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
    #         atten_output = tf.multiply(inputs, alpha)
    #         return atten_output
    #
    def gmp_AttentionLayer(self, inputs, name):
        # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为2×hidden_szie
            u_context = tf.Variable(tf.truncated_normal([self.config.num_filters]),
                                    name='u_context')
            # 使用一个全连接层编码GRU的输出的到期隐层表示,输出u的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.config.num_filters,
                                       activation_fn=tf.nn.tanh)
            print(u_context.shape, h.shape)
            # shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.multiply(h, u_context), dim=1)
            # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.multiply(inputs, alpha)
            return atten_output
###################################
    def ablstm(self):
        """ABLSTM模型"""
        # 词向量映射
        with tf.device('/cpu:0'):

            vocab_model = word2vec.Word2Vec.load('D:\\work\\Dail_NLP\\sentiment_analysis\\model\\segement_words_count_5_model_model1')
            # vocab_model = word2vec.Word2Vec.load('D:\\work\\Dail_NLP\\sentiment_analysis\\model\\sohu_single_word.model')

            embedding0 = []
            with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\word_segement_data\\vocab_count_5_segementword.txt', 'r', encoding='utf-8') as f:
                vocab = f.read().split('\n')
            for w in vocab:
                embedding0.append(vocab_model[w])
            emb1 = np.array(embedding0)
            embedding1 = tf.constant(value=emb1, dtype=tf.float32)
            print('W2V完成...')
            embedding_inputs = tf.nn.embedding_lookup(embedding1, self.input_x)
#############################################################################################################################
            BN_out_emb = tf.layers.batch_normalization(embedding_inputs, training=True)  # 增加的BN层

        #############################################################################################################################
        with tf.name_scope("ablstm"):
            # conv = tf.layers.conv1d(BN_out_emb, self.config.num_filters, self.config.kernel_size, name='conv')

            rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.config.num_filters),
                                    BasicLSTMCell(self.config.num_filters),
                                    inputs=BN_out_emb, dtype=tf.float32)
            fw_outputs, bw_outputs = rnn_outputs
            #####################################
            # attention layer
            W = tf.Variable(tf.random_normal([2*self.config.num_filters], stddev=0.1))
            # H = fw_outputs + bw_outputs  # (batch_size, seq_len, HIDDEN_SIZE)
            H = tf.concat((fw_outputs, bw_outputs), 2)

            H = tf.layers.batch_normalization(H, training=True)  # 增加的BN层


            M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
            self.alpha = tf.nn.softmax(tf.reshape(tf.matmul(tf.reshape(M, [-1, 2*self.config.num_filters]),
                                                            tf.reshape(W, [-1, 1])),
                                                  (-1, self.config.seq_length)))  # batch_size x seq_len
            r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                          tf.reshape(self.alpha, [-1, self.config.seq_length, 1]))

            r = tf.layers.batch_normalization(r, training=True)  # 增加的BN层
            r = tf.squeeze(r)
            h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE)
            # h_drop = tf.placeholder(tf.float32, [None, self.config.num_filters], name='keep_prob')
            h_drop = tf.nn.dropout(h_star, self.keep_prob)
            # print(h_drop.shape)
            #num_filters*(seq_len-kernel_size+1)
            # global max pooling layer
            # atten_conv = self.cove_AttentionLayer(conv, name='cove_attention')

########################################################################
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.layers.dense(tf.reshape(h_drop, [-1, 2*self.config.num_filters]), 2*self.config.num_filters, name='fc1')
            fc = tf.layers.batch_normalization(fc, training=True)#增加的BN层
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            #################################
            # FC_W = tf.Variable(tf.truncated_normal([self.config.num_filters, self.config.num_classes], stddev=0.1))
            # FC_b = tf.Variable(tf.constant(0., shape=[self.config.num_classes]))
            # self.logits = tf.nn.xw_plus_b(h_drop, FC_W, FC_b, name='fc2')
            #################################
            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
