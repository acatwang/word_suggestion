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


""" Viterbi
If we have a word sequence, what is the best tag sequence?

The method above lets us determine the probability for a single tag sequence.
But in order to find the best tag sequence, we need the probability for _all_ tag sequence.
What Viterbi gives us is just a good way of computing all those many probabilities
as fast as possible.
"""
distinct_tags = set(brown_tags)

# Given a sentence
sentence = ["I", "want", "to", "race"]
sentlen = len(sentence)

# viterbi: this is a list.
# for each step i in 1 .. sentlen,
# store a dictionary that maps each tag X
# to the probability of the best tag sequence of length i that ends in X
viterbi = []

# backpointer:
# for each step i in 1..sentlen,
# store a dictionary
# that maps each tag X
# to the previous tag in the best tag sequence of length i that ends in X
backpointer = []

##
# we first determine the viterby dictionary for the first word:
# For each tag, what is the probability of it following "START" and for it
# producing the first word of the sentence?
first_viterbi = { }
first_backpointer = { }
for tag in distinct_tags:
    # don't record anything for the START tag
    if tag == "START": continue

    first_viterbi[ tag ] = cpd_tags["START"].prob(tag) * cpd_tagwords[tag].prob( sentence[0] )
    first_backpointer[ tag ] = "START"

# store first_viterbi (the dictionary for the first word in the sentence)
# in the viterbi list, and record that the best previous tag
# for any first tag is "START"
viterbi.append(first_viterbi)
backpointer.append(first_backpointer)


# now we iterate over all remaining words in the sentence, from the second to the last.
for wordindex in range(1, len(sentence)):
         # start a new dictionary where we can store, for each tag, the probability
     # of the best tag sequence ending in that tag
         # for the current word in the sentence
    this_viterbi = { }
         # start a new dictionary we we can store, for each tag,
    # the best previous tag
    this_backpointer = { }
         # prev_viterbi is a dictionary that stores, for each tag, the probability
    # of the best tag sequence ending in that tag
    # for the previous word in the sentence.
    # So it stores, for each tag, the probability of a tag sequence up to the previous word
    # ending in that tag.
    prev_viterbi = viterbi[-1]

    # for each tag, determine what the best previous-tag is,
    # and what the probability is of the best tag sequence ending in this tag.
    # store this information in the dictionary this_viterbi
    for tag in distinct_tags:
        # don't record anything for the START tag
        if tag == "START": continue

        # if this tag is X and the current word is w, then
        # find the previous tag Y such that
        # the best tag sequence that ends in X
        # actually ends in Y X
        # that is, the Y that maximizes
        # prev_viterbi[ Y ] * P(X | Y) * P( w | X)
        # The following command has the same notation
        # that you saw in the sorted() command.
        best_previous = max(prev_viterbi.keys(),
                            key = lambda prevtag: \
            prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex]))

        # Instead, we can also use the following longer code:
        # best_previous = None
        # best_prob = 0.0
        # for prevtag in prev_viterbi.keys():
        #    prob = prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob(tag) * cpd_tagwords[tag].prob(sentence[wordindex])
        #    if prob >= best_prob:
        #        best_previous= prevtag
        #        best_prob = prob
        #
                  # this_viterbi[ tag ] is the probability of the best tag sequence ending in tag
        this_viterbi[ tag ] = prev_viterbi[ best_previous] * \
            cpd_tags[ best_previous ].prob(tag) * cpd_tagwords[ tag].prob(sentence[wordindex])
                 # this_backpointer[ tag ] is the most likely previous-tag for this current tag
        this_backpointer[ tag ] = best_previous

        # done with all tags in this iteration
    # so store the current viterbi step
    viterbi.append(this_viterbi)
    backpointer.append(this_backpointer)


# done with all words in the sentence.
# now find the probability of each tag
# to have "END" as the next tag,
# and use that to find the overall best sequence
prev_viterbi = viterbi[-1]
best_previous = max(prev_viterbi.keys(),
                    key = lambda prevtag: prev_viterbi[ prevtag ] * cpd_tags[prevtag].prob("END"))

prob_tagsequence = prev_viterbi[ best_previous ] * cpd_tags[ best_previous].prob("END")

# best tagsequence: we store this in reverse for now, will invert later
best_tagsequence = [ "END", best_previous ]
# invert the list of backpointers
backpointer.reverse()

# go backwards through the list of backpointers
# (or in this case forward, because we have inverter the backpointer list)
# in each case:
# the following best tag is the one listed under
# the backpointer for the current best tag
current_best_tag = best_previous
for bp in backpointer:
    best_tagsequence.append(bp[current_best_tag])
    current_best_tag = bp[current_best_tag]

best_tagsequence.reverse()
print "The sentence was:",
for w in sentence: print w,
print
print "The best tag sequence is:",
for t in best_tagsequence: print t,
print
print "The probability of the best tag sequence is:", prob_tagsequence