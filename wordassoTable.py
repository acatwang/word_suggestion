import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
from nltk.corpus import brown
import csv
import sys,string


bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
#tokenizer = RegexpTokenizer(r'\w+')

in_file = "res/sentences_train.csv"
tokens_all = []
tokens_add_all = []
with open(in_file, 'rU') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        sentence = row[0].decode('ascii', 'ignore')

        tokens = nltk.word_tokenize(sentence)
		# tokens = nltk.wordpunct_tokenize(row[0])
        tokens_add_all.append('START')
        for token in tokens:
            tokens_all.append(token)
            tokens_add_all.append(token)

        tokens_add_all.append('END')

# finder = BigramCollocationFinder.from_words(tokens_all)
# finder.apply_freq_filter(3)
# print finder.nbest(bigram_measures.pmi, 10)

print len(tokens_all)

#### Bigram Testing

finder = BigramCollocationFinder.from_words(tokens_all)
#finder.apply_freq_filter(1)
scored = finder.score_ngrams(bigram_measures.raw_freq)

#print scored
print [elem for elem in scored if elem[0]==('I','am')]

print [elem for elem in scored if elem[0]==('I','am')][0][1]

# print finder.ngram_fd.keys()
# cfd_words = nltk.ConditionalFreqDist(finder.ngram_fd.keys())
# cpd_words = nltk.ConditionalProbDist(cfd_words, nltk.MLEProbDist)
#
# print cfd_words
# print cpd_words

##### Trigram Testing
tri_finder = TrigramCollocationFinder.from_words(tokens_all)

# query_1, query_2 = query_sentence.split()[-2], query_sentence.split()[-1]
# finder.apply_ngram_filter(lambda w1, w2, w3: query_1 not in w1)
# finder.apply_ngram_filter(lambda w1, w2, w3: query_2 not in w2)
tri_scored = tri_finder.score_ngrams(trigram_measures.raw_freq)
# results = sorted(finder.nbest(trigram_measures.raw_freq, 10))
# finder = TrigramCollocationFinder.from_words(tokens_all)
# for w1, w2, w3 in results:
#     return w3
