__author__ = 'yywang'

import nltk
import codecs
import csv,json

word_tag_dict = {}
pos_dict = {}
with codecs.open("res/sentences_train.csv", "rU") as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        text = nltk.word_tokenize(row[0].decode('ascii', 'ignore'))
        tokens = nltk.pos_tag(text)

        for token in tokens:
            word_tag_dict[token[0]] = token[1]


#pos_dict.json = dict((value, key) for (key, value) in word_tag_dict.items())

for word, pos in word_tag_dict.items():
    if pos in pos_dict:
        if word not in pos_dict[pos]:
            pos_dict[pos][word] = 1
        else:
            pos_dict[pos][word] +=1
    else:
        pos_dict[pos] = {word: 1}


print pos_dict

with open("pos_dict_count.json",'wb') as f:
    json.dump(pos_dict, f)