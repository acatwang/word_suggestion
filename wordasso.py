import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.collocations import *
from nltk.corpus import brown
import csv


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

finder = BigramCollocationFinder.from_words(tokens_all)
finder.apply_freq_filter(3)
print finder.nbest(bigram_measures.pmi, 10)



