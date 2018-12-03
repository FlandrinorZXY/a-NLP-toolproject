import numpy as np
import pandas as pd
import os
base_dir = 'D:\\work\\Dail_NLP\\sentiment_analysis\\data\\'
train_dir = os.path.join(base_dir, 'dev_train.csv')
test_dir = os.path.join(base_dir, 'add_test.csv')
val_dir = os.path.join(base_dir, 'dev_val.csv')

train = pd.read_csv(train_dir, index_col=False)
test = pd.read_csv(test_dir, index_col=False)
val = pd.read_csv(val_dir, index_col=False)

def remove_yinhao(str):
    str = str.replace(' ', '')
    newstr = str[1:(len(str)-1)]
    return newstr

def get_wl(str):
    newstr = str[1:len(str)-1]
    templist = newstr.split(',')
    for i in range(len(templist)):
        templist[i] = remove_yinhao(templist[i])

    return templist

for i in range(train.shape[0]):
    print(i)
    train['content'][i] = get_wl(train['content'][i])

for i in range(test.shape[0]):
    print(i)
    test['content'][i] = get_wl(test['content'][i])

for i in range(val.shape[0]):
    print(i)
    val['content'][i] = get_wl(val['content'][i])

temp1 = []
temp2 = []
temp3 = []

for i in range(train.shape[0]):
    for x in train['content'][i]:
        temp1.append(x)

for i in range(test.shape[0]):
    for x in test['content'][i]:
        temp2.append(x)

for i in range(val.shape[0]):
    for x in val['content'][i]:
        temp3.append(x)

temp = temp1+temp2+temp3
list2 = []

for i in range(len(temp)):
    if i % 1000 ==0:
        print(i)
    if temp[i] not in list2:
        list2.append(temp[i])

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\re_vocab.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(list2) + '\n')
f.close()
# vocab = list(set(temp))
# vocab.sort(key=temp.index)
#####################################

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\re_vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read().split('\n')

vocab.append('<PAD>')
from gensim.models import word2vec
re_model = word2vec.Word2Vec([vocab], size=256, window=5, min_count=1, workers=4)
re_model.save('D:\\work\\Dail_NLP\\sentiment_analysis\\model\\re_model')