import pandas as pd
import numpy as np
import time
add_data = pd.read_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\val_13.csv',encoding='utf-8').astype(str)

with open('vocab1.txt', 'r', encoding='utf-8') as f:
    og_vocab = f.read().split('\n')

demo_length = add_data.shape[0]
source0 = add_data.iloc[0:demo_length, :]
import jieba
title = list(source0['title'])
content = list(source0['content'])
N = 11000
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


print(len(og_vocab), len(new_vocab))

vocab = og_vocab+new_vocab
vocab = list(set(vocab))
print(len(vocab))

with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\add_vocab1.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(vocab) + '\n')
f.close()

word_df = pd.DataFrame()
word_df['og_title'] = title_words_list
word_df['og_content'] = content_words_list
word_df['content'] = word_df['og_title'] + word_df['og_content']
word_df = word_df.drop(['og_content'], axis=1)
word_df = word_df.drop(['og_title'], axis=1)
word_df['label'] = source0['properties']

word_df.to_csv('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\add_test1.csv',index=False)