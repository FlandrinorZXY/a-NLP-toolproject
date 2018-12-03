import pandas as pd
import numpy as np
import time

source_path = 'D:\\work\\Dail_NLP\\sentiment_analysis\\sentiment_train_0823.csv'
with open(source_path) as f:
    source0 = pd.read_csv(source_path, encoding='utf-8', index_col=0)

demo_length = source0.shape[0]
source0 = source0.iloc[0:demo_length, :]
import jieba
title = list(source0['title'])
content = list(source0['content'])
N = 1500
import jieba.posseg as psg

stopwords = ['', '。', '，', '（', '）', '&', '@', '#', '￥', '%', '^', '/',
             '、', '：', ' ；', '(', ')', ',', '.', '“', '”', '：', '；', '-', '+', ':', '"', ';', ' ', ' ', '{', '}', '||', '\\'
             '>', '<', '》', '《', '[', ']', '【', '】', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_words(sentence):
    sent = []
    for i in sentence:
        sent.append(i)
    return sent

def get_useful_words(sentences,stopwords):
    sent_words = []
    for i in range(len(sentences)):
        if (sentences[i]in stopwords):
            sentences[i] = 'ULW'
    for i in range(len(sentences)):
        sent_words.append(sentences[i])
    return sent_words

title_words_list = []
content_words_list = []
vocab = []
start = time.time()
for i in range(len(title)):
    if i % 100 == 0:
        print(i)

    sent_words_title = get_useful_words(get_words(title[i]), stopwords=stopwords)
    sent_words_content = get_useful_words(get_words(content[i]), stopwords=stopwords)
    vocab = vocab+sent_words_content+sent_words_title
    title_words_list.append(sent_words_title)
    content_words_list.append(sent_words_content)

print('step1:', time.time()-start)


title_temp=[]
content_temp=[]
for i in range(len(title_words_list)):
    title_temp.append(get_useful_words(title_words_list[i], stopwords=stopwords))
    content_temp.append(get_useful_words(content_words_list[i], stopwords=stopwords))

new_vocab = get_useful_words(vocab,stopwords=stopwords)
df_vocab = pd.DataFrame()
df_vocab['vocab'] = new_vocab
a = pd.DataFrame(df_vocab['vocab'].value_counts()).reset_index()
a.columns = ['vocab','vocab_counts']
final_df_vocab = pd.DataFrame(a['vocab'][a['vocab_counts']>= 5])

vocab_count_5 = list(final_df_vocab['vocab'])


final_title_temp=[]
final_content_temp=[]

def get_useful_words_1(sentences,vocab):
    sent_words = []
    for i in range(len(sentences)):
        if (sentences[i] not in vocab):
            sentences[i] = 'ULW'
    for i in range(len(sentences)):
        sent_words.append(sentences[i])
    return sent_words

for i in range(len(title_words_list)):
    if i % 100 == 0:
        print(i)
    final_title_temp.append(get_useful_words_1(title_temp[i], vocab=vocab_count_5))
    final_content_temp.append(get_useful_words_1(content_temp[i], vocab=vocab_count_5))

final_data = pd.DataFrame()
final_data['title'] = final_title_temp
final_data['content'] = final_content_temp
final_data['properties'] = source0['properties']

final_data.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\my_single_words_data.csv',encoding='utf-8', index=False)
with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\vocab_count_5_singleword.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(vocab_count_5) + '\n')
f.close()

#清洗初始vocab用于训练词向量
model_vocab_temp=[]
for i in range(len(vocab)):
    if i % 100 == 0:
        print(i)
    if vocab[i] not in vocab_count_5:
        model_vocab_temp.append('ULW')
    if vocab[i] in vocab_count_5:
        model_vocab_temp.append(vocab[i])
with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\model_vocab_count_5_singleword.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(model_vocab_temp) + '\n')
f.close()

model_vocab_temp1 = model_vocab_temp+['']

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\model_vocab_count_5_singleword1.txt', 'w', encoding='utf-8') as f1:
    f1.write('\n'.join(model_vocab_temp1) + '\n')
f.close()

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\model_vocab_count_5_singleword.txt', 'r', encoding='utf-8') as f1:
    model_vocab_temp1 = f1.readlines()
f1.close()

model_vocab_temp=[]
for i in range(len(model_vocab_temp1)):
    model_vocab_temp.append(model_vocab_temp1[i].strip('\n'))
model_vocab_temp = model_vocab_temp+['']
from gensim.models import word2vec
single_words_count_5_model1 = word2vec.Word2Vec([model_vocab_temp], size=256, window=5, min_count=1, workers=4)
single_words_count_5_model1.save('D:\\work\\Dail_NLP\\sentiment_analysis\\model\\single_words_count_5_model_model1')
##############################数据切分
final_data['full_content'] = final_data['title']+final_data['content']
final_data['label'] = final_data['properties']

from sklearn.model_selection import train_test_split

X, y = final_data['full_content'], final_data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)#随机选择25%作为测试集，剩余作为训练集

train = pd.DataFrame()
train['content'] = X_train
train['label'] = y_train
train = train.reset_index()
train = train.drop(['index'], axis=1)

test = pd.DataFrame()
test['content'] = X_test
test['label'] = y_test
test = test.reset_index()
test = test.drop(['index'], axis=1)

train_train,train_val,train_label,val_label = train_test_split(train['content'], train['label'],test_size=0.25,random_state=0)

the_train = pd.DataFrame()
the_train['content'] = train_train
the_train['label'] = train_label
the_train = the_train.reset_index()
the_train = the_train.drop(['index'], axis=1)

val = pd.DataFrame()
val['content'] = train_val
val['label'] = val_label
val = val.reset_index()
val = val.drop(['index'], axis=1)

the_train.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\TEXTCNN_SINGLEWORDS\\single_words_train.csv', encoding='utf-8', index=False)
test.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\TEXTCNN_SINGLEWORDS\\single_words_test.csv', encoding='utf-8', index=False)
val.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\TEXTCNN_SINGLEWORDS\\single_words_val.csv', encoding='utf-8', index=False)


zz = pd.read_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\TEXTCNN_SINGLEWORDS\\single_words_val.csv', index_col=False)

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\vocab_count_5_singleword.txt', 'r', encoding='utf-8') as f:
    zzz = f.read().split('\n')