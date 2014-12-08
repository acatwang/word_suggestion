import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
from nltk.corpus import brown
import csv
import sys

spam = sys.argv[1]

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()
tokenizer = RegexpTokenizer(r'\w+')

in_file = "res/sentences_train.csv"
tokens_all = []
with open(in_file, 'rU') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		tokens = tokenizer.tokenize(row[0])
		# tokens = nltk.wordpunct_tokenize(row[0])
		for token in tokens:
			tokens_all.append(token)

# finder = BigramCollocationFinder.from_words(tokens_all)
# finder.apply_freq_filter(3)
# print finder.nbest(bigram_measures.pmi, 10)

#### Bigram Testing
if spam == '2':
	finder = BigramCollocationFinder.from_words(tokens_all)
	more = True
	while more:
		query_sentence = raw_input('Enter a sentence: ')
		if query_sentence.lower() == 'no more':
			print 'Good Bye!'
			more = False
		else:
			query = query_sentence.split()[-1]
			finder.apply_ngram_filter(lambda w1, w2: query != w1)
			scored = finder.score_ngrams(bigram_measures.raw_freq)
			results = sorted(finder.nbest(bigram_measures.raw_freq, 10))
			finder = BigramCollocationFinder.from_words(tokens_all)
			for w1, w2 in results:
				print w2


##### Trigram Testing
if spam == '3':
	finder = TrigramCollocationFinder.from_words(tokens_all)
	more = True
	while more:
		query_sentence = raw_input('Enter a sentence: ')
		if query_sentence.lower() == 'no more':
			print 'Good Bye!'
			more = False
		else:
			query_1, query_2 = query_sentence.split()[-2], query_sentence.split()[-1]
			finder.apply_ngram_filter(lambda w1, w2, w3: query_1 != w1)
			finder.apply_ngram_filter(lambda w1, w2, w3: query_2 != w2)
			scored = finder.score_ngrams(trigram_measures.raw_freq)
			results = sorted(finder.nbest(trigram_measures.raw_freq, 10))
			finder = TrigramCollocationFinder.from_words(tokens_all)
			for w1, w2, w3 in results:
				print w3
