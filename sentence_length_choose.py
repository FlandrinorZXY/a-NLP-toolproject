import numpy as np
import pandas as pd
import os
base_dir = 'D:\\work\\Dail_NLP\\sentiment_analysis\\data\\'
train_dir = os.path.join(base_dir, 'dev_train.csv')
test_dir = os.path.join(base_dir, 'test.csv')
val_dir = os.path.join(base_dir, 'dev_val.csv')
vocab_dir = os.path.join(base_dir, 'vocab.txt')

# base_dir = 'D:\\work\\Dail_NLP\\sentiment_analysis\\data\\'
# train_dir = os.path.join(base_dir, 'dev_train.csv')
# test_dir = os.path.join(base_dir, 'add_test.csv')
# val_dir = os.path.join(base_dir, 'dev_val.csv')
# vocab_dir = os.path.join(base_dir, 'add_vocab.txt')
train_data = pd.read_csv(train_dir, index_col=False)
test_data = pd.read_csv(test_dir, index_col=False)
val_data = pd.read_csv(val_dir, index_col=False)
###############################################
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

for i in range(train_data.shape[0]):
    print(i)
    train_data['content'][i] = get_wl(train_data['content'][i])

for i in range(test_data.shape[0]):
    print(i)
    test_data['content'][i] = get_wl(test_data['content'][i])

for i in range(val_data.shape[0]):
    print(i)
    val_data['content'][i] = get_wl(val_data['content'][i])

temp1 = []
temp2 = []
temp3 = []

for i in range(train_data.shape[0]):
    for x in train_data['content'][i]:
        temp1.append(x)

for i in range(test_data.shape[0]):
    for x in test_data['content'][i]:
        temp2.append(x)

for i in range(val_data.shape[0]):
    for x in val_data['content'][i]:
        temp3.append(x)

vocab = list(set(temp1+temp2+temp3))

with open('vocab1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(vocab) + '\n')
f.close()

train_data.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\train_data1.csv',index=False)
test_data.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\test_data1.csv',index=False)
val_data.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\val_data1.csv',index=False)

##############################################
def get_length(data):
    temp = []
    for i in range(data.shape[0]):
        temp.append(len(data['content'][i]))
    result = pd.value_counts(temp)
    return temp, result

train_lenlist, train_r = get_length(train_data)
test_lenlist, test_r = get_length(test_data)
val_lenlist, val_r = get_length(val_data)
train_data['length'] = train_lenlist
print(train_r)

import matplotlib.pyplot as plt

plt.hist(train_lenlist, bins=80)

tryl = train_data['content'][0]
#####################################################



