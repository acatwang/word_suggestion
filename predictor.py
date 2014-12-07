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

data =[]
punctuation = "!\"#%&'()*+,-./:;<=>?@[\]^_`{|}~"

"""
Tagging part of speech
Use maxent treebank pos tagging model in NLTK by default
Each consisting of a list of tokens
"""

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
        data.append(simplified_tag)
print data[0:10]

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tag for word, tag in data[0])
print finder.nbest(bigram_measures, 5)
"""

# Ref http://www.katrinerk.com/courses/python-worksheets/hidden-markov-models-for-pos-tagging-in-python
# It estimates the probability of a future tag sequence for a given word/tag sequence as follows:
#
# Say words = w1....wN
# and states = s1 ... sN
#
# then
# P(state | words,states) is_proportional_to  product P(si | s{i-1}) P(wi | si)
#
# To find the best tag sequence for a given sequence of words,
# we want to find the tag sequence that has the maximum P(state | words)

"""Part 1 : P(wi | si)
Estimating P(wi | si) from corpus data using Maximum Likelihood Estimation (MLE):
P(wi | si) = count(wi, si) / count(si)

We add an artificial "start" tag at the beginning of each sentence, and
We add an artificial "end" tag at the end of each sentence.
So we start out with the brown tagged sentences,
add the two artificial tags,
and then make one long list of all the tag/word pairs.
"""
brown_tags_words = [ ]
for sent in brown.tagged_sents(tagset="universal"):
    # sent is a list of word/tag pairs
    # add START/START at the beginning
    brown_tags_words.append( ("START", "START") )
    # then all the tag/word pairs for the word/tag pairs in the sentence
    brown_tags_words.extend([ (tag, word) for (word, tag) in sent ])
    # then END/END
    brown_tags_words.append( ("END", "END") )

# conditional frequency distribution
cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
# conditional probability distribution
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)

print "The probability of an adjective (ADJ) being 'new' is", cpd_tagwords["ADJ"].prob("new")
print "The probability of an adjective (ADJ) being 'new' is", cpd_tagwords["ADJ"].prob("new")

"""Part 2 : P( si | s{i-1})
Estimating P(ti | t{i-1}) from corpus data using Maximum Likelihood Estimation (MLE):
P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
"""

brown_tags = [tag for (tag, word) in brown_tags_words]

# make conditional frequency distribution:
# count(t{i-1} ti)
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
# make conditional probability distribution, using
# maximum likelihood estimate:
# P(ti | t{i-1})
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
print cpd_tags
print "If we have just seen 'DET', the probability of 'NOUN' is", cpd_tags["DET"].prob("NOUN")

""" putting things together:
what is the probability of the tag sequence "PRO V TO V" for the word sequence "I want to race"?
It is
P(START) * P(PRO|START) * P(I | PRO) *
           P(V | PRO) * P(want | V) *
           P(TO | V) * P(to | TO) *
           P(VB | TO) * P(race | V) *
           P(END | V)
"""
prob_tagsequence = cpd_tags["START"].prob("PRON") * cpd_tagwords["PRON"].prob("I") * \
    cpd_tags["PRON"].prob("VERB") * cpd_tagwords["VERB"].prob("want") * \
    cpd_tags["VERB"].prob("ADP") * cpd_tagwords["ADP"].prob("to") * \
    cpd_tags["ADP"].prob("VERB") * cpd_tagwords["VERB"].prob("race") * \
    cpd_tags["VERB"].prob("END")

print cpd_tags["START"].prob("PRON")
print cpd_tags["PRON"].prob("VERB")
print "The probability of the tag sequence 'START PRO V TO V END' for 'I want to race' is:", prob_tagsequence


