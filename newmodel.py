vec_dir ='D:\\work\\Dail_NLP\\sentiment_analysis\\data\\cc.zh.300.vec\\cc.zh.300.vec'
with open(vec_dir, 'r', encoding='utf-8') as f:
    content = f.readlines()

vocab = []
vec = []
for sentence in content:
    word, vector = sentence.split(' ', 1)
    vocab.append(word)
    vec.append(vector)
# with open('D:\\work\\Dail_NLP\\sentiment_analysis\\data\\big_vocab1.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(vocab) + '\n')
# f.close()
vecnew = []
for w in vec:
    a = w.strip('\n').split(' ')
    temp = []
    for s in a:
        temp.append(float(s))
    vecnew.append(temp)