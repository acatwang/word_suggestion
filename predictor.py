__author__ = 'Yi-Yin wang'
import csv, string
import codecs
import numpy as np
import nltk
from sklearn import hmm
#from nltk.tag import hmm
from nltk.corpus import brown
from nltk.tag.mapping import map_tag
from nltk.collocations import *
#from hiddenState import transMat, tagset, tagset_dict

tokens =[]
punctuation = "!\"#%&'()*+,-./:;<=>?@[\]^_`{|}~"

"""
Tagging part of speech
Use maxent treebank pos tagging model in NLTK by default
Each consisting of a list of tokens
"""


print string.punctuation
with codecs.open("res/sentences_train.csv", "rU") as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        # remove punctuation
        # Ignore ascii decode error
        sentence = row[0].translate(string.maketrans("", ""), punctuation).decode('ascii', 'ignore')
        text = nltk.word_tokenize(sentence)
        original_tag = nltk.pos_tag(text)
        simplified_tag = [(word, map_tag('en-ptb','universal', tag)) for word, tag in original_tag] # Map original Map to universla tags
        simplified_tag = [(u'START',u'START')] + simplified_tag + [(u'END',u'END')]  # manually add two tags
        #tokens.append(simplified_tag)
        tokens.extend(simplified_tag)
print tokens[0:10]


# TODO:BenchMark
# Using word association to generate suggestion
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(word for word,tag in tokens)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print sorted(bigram for bigram, score in scored)

finder.apply_freq_filter(2)
scored = finder.score_ngrams(bigram_measures.raw_freq)
print sorted(bigram for bigram, score in scored)


#finder.nbest(bigram_measures, 5)

