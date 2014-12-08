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

for key, value in word_tag_dict.items():
    if value in pos_dict:
        if key not in pos_dict[value]:
            pos_dict[value].append(key)
    else:
        pos_dict[value] = []
        pos_dict[value].append(key)

print pos_dict

with open("pos_dict.json",'wb') as f:
    json.dump(pos_dict, f)