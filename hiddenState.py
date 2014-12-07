__author__ = 'yywang'

import nltk
from nltk.collocations import *
import numpy as np
from sklearn import hmm
from nltk.corpus import brown
from hmmpytk import hmm_faster
import math
# Use corpus brown to train HMM model

# Generate start probability and transition matrix using brown corpus
brown_tagged = brown.tagged_words('ca01',tagset="universal")
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(t for w, t in brown_tagged)

print finder.nbest(bigram_measures.pmi,10)
print finder.score_ngrams(bigram_measures.raw_freq)
#print finder.score_ngrams(bigram_measures.raw_freq)
tag_fd = nltk.FreqDist(tag for (word,tag) in brown_tagged)
pairfreq_Dict = dict(finder.score_ngrams(bigram_measures.raw_freq))


freqTable = np.array(pairfreq_Dict.values())
freqTable = np.around(freqTable,decimals=4)
print sum(freqTable)

print pairfreq_Dict[(u'DET',u'.')]
print pairfreq_Dict[(u'.', u'DET')]


def make_transmat_prior(start_prob):
    mat = []
    for x in tagset:
        print x
        row =[]
        for y in tagset:
            try:
                # print pairfreq_Dict[(x, y)]
                row.append(round(pairfreq_Dict[(x, y)],4))
            except KeyError:
                #print "No freq"
                row.append(0)

        row =[elem/start_prob[tagset.index(x)] for elem in row]

        print sum(row)

        num_of_zero_col = [idx for idx,val in enumerate(row) if val==0]
        print num_of_zero_col

        # impute a small number to avoid error in log
        for i in num_of_zero_col:
            row[i] = 0.00000001
        print row
        diff = 1 - sum(row)
        if sum(row) >= 1:
            bigger_freq = [row.index(f) for f in row if f>abs(diff)]
            print bigger_freq
            for elem in bigger_freq:
                row[elem]+= (diff/len(bigger_freq))

            #print row
            print sum(row)

        elif sum(row) < 1:

            if len(num_of_zero_col) == 0:
                row = [x+(diff/11) for x in row]
                print sum(row)
            else: # impute zero occurrence
                for x in num_of_zero_col:
                    row[x] += diff/len(num_of_zero_col)
                print sum(row)


        mat.append(row)

    return np.matrix(mat)


# States - Universal Tagset
print tag_fd.most_common(11)
tagset = [tag for (tag, freq) in tag_fd.most_common(11)]
start_prob = np.array([freq/float(len(brown_tagged)) for (tag, freq) in tag_fd.most_common(11)])
start_prob = np.around(start_prob, decimals=4)
trans_matrix = make_transmat_prior(start_prob)
print start_prob

hmmModel = hmm._BaseHMM(11,start_prob,trans_matrix)
print tagset
tagset_dict = dict(((v,i) for i,v in enumerate(tagset)))
transMat = hmmModel.transmat_


#hmmModel.predict([0,1], algorithm="viterbi")


""" HMM Sample code
startprob = np.array([0.6, 0.3, 0.1])
transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
means = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
covars = np.tile(np.identity(2), (3, 1, 1))
model = hmm.GaussianHMM(3, "full", startprob, transmat)
model.means_ = means
model.covars_ = covars
X, Z = model.sample(100)

model2 = hmm.GaussianHMM(3, "full")
model2.fit([X])
Z2 = model2.predict(X)

print X
print Z2

print len(X)
print len(Z2)
"""