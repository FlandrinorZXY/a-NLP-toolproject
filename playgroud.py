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
             '、', '：', ' ；', '(', ')', ',', '.', '“', '”', '：', '；', '-', '+', ':', '"', ';', ' ', ' ',
             '>', '<', '》', '《']
def get_useful_words(sentences,unkeepwords,stopwords):
    sent_words = []
    for i in range(len(sentences)):
        if (sentences[i].flag in unkeepwords) | (sentences[i].word in stopwords):
            sentences[i].word = 'ULW'
    for i in range(len(sentences)):
        sent_words.append(sentences[i].word)
    return sent_words

title_words_list = []
content_words_list = []
vocab = []
start = time.time()
for i in range(len(title)):
    sent_words_title = get_useful_words(psg.lcut(title[i]), unkeepwords=['m', 'eng'], stopwords=stopwords)
    sent_words_content = get_useful_words(psg.lcut(content[i]), unkeepwords=['m', 'eng'], stopwords=stopwords)
    vocab=vocab+sent_words_content+sent_words_title
    title_words_list.append(sent_words_title)
    content_words_list.append(sent_words_content)

# vocab = set(vocab)
result_vocab = pd.DataFrame(pd.value_counts(vocab)).reset_index()
result_vocab.columns = ['words', 'count']
print('获得vocab并将title、content分词,用时：', time.time()-start)
print('构造映射词典...')
#将vocab中频次不小于5的词构建词典/前1500
# over5_vocab = result_vocab[result_vocab['count'].values >= 5]
over5_vocab = result_vocab.iloc[0:N, :]
new_vocab = list(over5_vocab['words'])
vocab_to_int = {word: idx for idx, word in enumerate(new_vocab)}
int_to_vocab = {idx: word for idx, word in enumerate(new_vocab)}

def vocab_transTo_int(sentence,vocab_to_int,vocab_bag):
    temp = []
    for i in range(len(sentence)):
        if sentence[i] in vocab_bag:
            temp.append(vocab_to_int.__getitem__(sentence[i]))
    return temp


title_int_list = []
content_int_list = []
for i in range(len(title_words_list)):
    sent_int_title = vocab_transTo_int(title_words_list[i], vocab_to_int,new_vocab)
    sent_int_content =vocab_transTo_int(content_words_list[i], vocab_to_int,new_vocab)
    title_int_list.append(sent_int_title)
    content_int_list.append(sent_int_content)


df = pd.DataFrame()
df['title'] = title_int_list
df['content'] = content_int_list
df['label'] = source0['properties']
vocab_size = len(new_vocab)

with open('vocab.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_vocab) + '\n')
f.close()

new_df = pd.DataFrame()
new_df['content'] = df['title']+df['content']
new_df['label'] = df['label']
new_df.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\数据\\all_int_data.csv', index=False)

word_df = pd.DataFrame()
word_df['og_title'] = title_words_list
word_df['og_content'] = content_words_list
word_df['content'] = word_df['og_title'] + word_df['og_content']
word_df = word_df.drop(['og_content'], axis=1)
word_df = word_df.drop(['og_title'], axis=1)
word_df['label'] = df['label']

word_df.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\数据\\all_words_data.csv', index=False)
##################################################################


test = word_df.iloc[0:2000, :]
dev_source = word_df.iloc[2000:source0.shape[0], :].reset_index()
dev_source = dev_source.drop(['index'], axis=1)

Y = dev_source['label']
dev_source = dev_source.drop(['label'], axis=1)
X = dev_source


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=33)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

dev_train = pd.DataFrame()
dev_train['content'] = X_train['content']
dev_train['label'] = Y_train
dev_train = dev_train.reset_index().drop(['index'], axis=1)

dev_val = pd.DataFrame()
dev_val['content'] = X_test['content']
dev_val['label'] = Y_test
dev_val = dev_val.reset_index().drop(['index'], axis=1)

dev_train.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\数据\\dev_train.csv', index=False)
dev_val.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\数据\\dev_val.csv', index=False)
test.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\test.csv', index=False)


def max_l(data):
    l = 0
    num = 0
    for i in range(data.shape[0]):
        if len(data['content'][i]) > l:
            l = len(data['content'][i])
            num = i
    return num, l

num,l = max_l(word_df)

try_data = pd.read_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\dev_val.csv', encoding='utf-8', index_col=False)