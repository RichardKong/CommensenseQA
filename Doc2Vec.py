# coding:utf-8
import json
import sys
import gensim
import sklearn
import numpy as np

from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from scipy.spatial import distance

TaggededDocument = gensim.models.doc2vec.TaggedDocument
import jieba

f1 = open("./train_easy.txt", "r", encoding='utf-8', errors='ignore')
f2 = open("./train_easy_copy.txt", 'w', encoding='utf-8', errors='ignore')

lines = f1.readlines()  # 读取全部内容
w = ''
# 对句子分词且将分词后的文本存入txt
for line in lines:
    line.replace('\t', '').replace('\n', '').replace(' ', '')
    seg_list = jieba.cut(line, cut_all=False)
    f2.write(" ".join(seg_list))
f1.close()
f2.close()


def get_datasest():
    with open(u"./train_easy_copy.txt", 'r', encoding='utf-8', errors='ignore') as cf:
        docs = cf.readlines()
        print(len(docs))

    x_train = []

    # 将分词后的文本给予索引
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l - 1] = word_list[l - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train


# 获取语料库中的向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)


# 训练模型
def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('model_dm')
    print(model_dm)
    return model_dm


# 测试模型
def test():
    model_dm = Doc2Vec.load("model_dm")
    test_text = ["like","beauty"]
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print("test\n",inferred_vector_dm, "\n")
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)


    return sims
"""
def avg_feature_vector(sentence, model, index2word_set):
        feature = np.zeros((200, ), dtype='float32')
        n_word = 0
        words = sentence.split(' ')
        for word in words:
            if word in index2word_set:
                n_word += 1
                feature = np.add(feature, model[word])
        if n_word > 0:
            feature = np.divide(feature, n_word)

        return feature
"""


def compare_vectors(vector1, vector2):
    cos_distances = []
    for i in range(len(vector1)):
        d = distance.cosine(vector1[i], vector2[i])
        cos_distances.append(d)
    print(np.median(cos_distances))
    print(np.std(cos_distances))

def test2():
    words = []
    choices = []
    answers = []
    questions = []
    model_dm = Doc2Vec.load("model_dm")
    f = open("I:\\19-20\机器学习\data\\train_rand_split_EASY.jsonl", 'r')
    for line in f.readlines():
        dic = json.loads(line)
        question = dic['question']['stem']
        choose = dic['question']['choices']
        answer = dic['answerKey']
        for i in range(5):
            choice = choose[i]['text']
            choices.append(choice)
            #words.append(choice.split(' '))
        questions.append(question)
        answers.append(answer)
    #print(answers)

    from scipy import spatial

    right = 0
    for i in range(5):
        sims = []
        array_q = []
        #print(questions[i])
        array_q.append(questions[i])
        #print("question", array_q)
        question_feature = model_dm.infer_vector(array_q)
        question_feature = question_feature.flatten()
        #print("question_feature\n", question_feature)
        for j in range(5):
            #print(i * 5 + j)
            #print(choices[i * 5 + j])
            array_a = []
            array_a.append(choices[i * 5 + j])
            model_dm = Doc2Vec.load("model_dm")
            choice_feature = model_dm.infer_vector(array_a)
            choice_feature = choice_feature.flatten()
            #print("type\n", type(choice_feature), type(model_dm.docvecs), "\n")
            #sim = model_dm.docvecs.distance(question_feature,choice_feature)
            sim = 1 - spatial.distance.cosine(question_feature, choice_feature)
            sims.append(sim)
        print("A: ",sims[0], "\nB: ",sims[1], "\nC: ",sims[2], "\nD: ",sims[3], "\nE: ",sims[4], "\n")
        answer = sims.index(max(sims))
        if answer == 0:
            answer = 'A'
        elif answer == 1:
            answer = 'B'
        elif answer == 2:
            answer = 'C'
        elif answer == 3:
            answer = 'D'
        elif answer == 4:
            answer = 'E'
        if answer == answers[i]:
            right +=1

    print("number of right answer",right)


# 主函数应用-
if __name__ == '__main__':

    x_train = get_datasest()
    #model_dm = train(x_train)
    sims = test()
    print("sims\n",sims,"\n")
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print(words, "\n", count, sim, len(sentence[0]))

    test2()