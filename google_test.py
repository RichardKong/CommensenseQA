from gensim.models import word2vec
import json
import gensim

filepath = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filepath, binary=True)

import numpy as np


def avg_feature_vector(sentence, model):
    feature = np.zeros((300,), dtype='float32')
    n_word = 0
    words = sentence.split(' ')
    for word in words:
        if word in model:
            n_word += 1
            feature = np.add(feature, model[word])
    if n_word > 0:
        feature = np.divide(feature, n_word)

    return feature


def deal_json(filename, s_number):
    f = open(filename, 'r')
    questions = []
    choices = []
    answers = []
    i = 0

    for line in f.readlines():
        if i >= s_number:
            break

        d_choices = []
        dic = json.loads(line)
        answers.append(dic['answerKey'])
        question = dic['question']['stem']
        question = question.replace(',', '')
        question = question.lower()
        question = question.replace('?', '')
        questions.append(question)
        a = dic['question']['choices']
        for j in range(5):
            d_choices.append(a[j]['text'])
        choices.append(d_choices)
        i += 1
    return questions, choices, answers


filename_2 = 'I://19-20//机器学习//data/train_rand_split_EASY.jsonl'
line_number = 9740
question, choices, answers = deal_json(filename_2, line_number)

# for line in f.readlines():
#     dic = json.loads(line)
#     word = dic['question']['stem']
#     word = word.split(' ')
#     words.append(word)
#     choose = dic['question']['choices']
#     for i in range(5):
#         choice = choose[i]['text']
#     words.append(choice.split(' '))

from scipy import spatial

#print(question)
right = 0
for i in range(9740):
    sims = []
    question_feature = avg_feature_vector(question[i], model)
    for j in range(5):
        choice_feature = avg_feature_vector(choices[i][j], model)
        sim = 1 - spatial.distance.cosine(question_feature, choice_feature)
        sims.append(sim)
        #print(sim)

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
        right += 1

#print('Answer:', answer)
print('right', right)