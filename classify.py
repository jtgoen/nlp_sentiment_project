import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import MaxentClassifier
import random
import json
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import stem

def sanitizeWord(word):
    puncSet = set(['.',',','(',')','?','!',':'])
    for punc in puncSet:
        word = word.replace(punc, "")
    word = word.lower()
    return word

stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))

reviews_1 = []
for line in open('reviews_1.json'):
    words = ""
    for word in json.loads(line)['text'].split():
    	sanitized_word = sanitizeWord(word)
    	if sanitized_word not in stopset:
        	words = words + sanitized_word + " "
    reviews_1.append(words)		

reviews_2 = []
for line in open('reviews_2.json'):
    words = ""
    for word in json.loads(line)['text'].split():
    	sanitized_word = sanitizeWord(word)
    	if sanitized_word not in stopset:
        	words = words + sanitized_word + " "
    reviews_2.append(words)		

reviews_3 = []
for line in open('reviews_3.json'):
    words = ""
    for word in json.loads(line)['text'].split():
    	sanitized_word = sanitizeWord(word)
    	if sanitized_word not in stopset:
        	words = words + sanitized_word + " "
    reviews_3.append(words)		

reviews_4 = []
for line in open('reviews_4.json'):
    words = ""
    for word in json.loads(line)['text'].split():
    	sanitized_word = sanitizeWord(word)
    	if sanitized_word not in stopset:
        	words = words + sanitized_word + " "
    reviews_4.append(words)		
 
reviews_5 = []
for line in open('reviews_5.json'):
    words = ""
    for word in json.loads(line)['text'].split():
    	sanitized_word = sanitizeWord(word)
    	if sanitized_word not in stopset:
        	words = words + sanitized_word + " "
    reviews_5.append(words)		
 
def word_split(data):	
	data_new = []
	for word in data:
		word_filter = [i.lower() for i in word.split()]
		data_new.append(word_filter)
	return data_new
 
# def word_split_sentiment(data):
# 	data_new = []
# 	for (word, sentiment) in data:
# 		word_filter = [i.lower() for i in word.split()]
# 		data_new.append((word_filter, sentiment))
# 	return data_new
	
def word_feats(words):	
	return dict([(word, True) for word in words])

def stemmed_word_feats(words):
	porter_stemmer = stem.porter.PorterStemmer()
	return dict([(porter_stemmer.stem(word), True) for word in words])
      
# def stopword_filtered_word_feats(words):
#     return dict([(word, True) for word in words if word not in stopset])
 
# def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     """
#     print words
#     for ngram in itertools.chain(words, bigrams): 
# 		if ngram not in stopset: 
# 			print ngram
#     exit()
#     """    
#     return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
    
# def bigram_word_feats_stopwords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
#     bigram_finder = BigramCollocationFinder.from_words(words)
#     bigrams = bigram_finder.nbest(score_fn, n)
#     """
#     print words
#     for ngram in itertools.chain(words, bigrams): 
# 		if ngram not in stopset: 
# 			print ngram
#     exit()
#     """    
#     return dict([(ngram, True) for ngram in itertools.chain(words, bigrams) if ngram not in stopset])
 
# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx):
	
	one_star_feats = [(featx(f), '1') for f in word_split(reviews_1)]
	two_star_feats = [(featx(f), '2') for f in word_split(reviews_2)]
	three_star_feats = [(featx(f), '3') for f in word_split(reviews_3)]
	four_star_feats = [(featx(f), '4') for f in word_split(reviews_4)]
	five_star_feats = [(featx(f), '5') for f in word_split(reviews_5)]
	    
	one_star_cutoff = len(one_star_feats)*3/4
	two_star_cutoff = len(two_star_feats)*3/4
	three_star_cutoff = len(three_star_feats)*3/4
	four_star_cutoff = len(four_star_feats)*3/4
	five_star_cutoff = len(five_star_feats)*3/4
 
	trainfeats = one_star_feats[:one_star_cutoff] + two_star_feats[:two_star_cutoff] + three_star_feats[:three_star_cutoff] + four_star_feats[:four_star_cutoff] + five_star_feats[:five_star_cutoff]
	testfeats = one_star_feats[one_star_cutoff:] + two_star_feats[two_star_cutoff:] + three_star_feats[three_star_cutoff:] + four_star_feats[four_star_cutoff:] + five_star_feats[five_star_cutoff:]


	classifierName = 'Maximum Entropy'
	classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)

		
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testfeats):
			refsets[label].add(i)
			observed = classifier.classify(feats)
			testsets[observed].add(i)

	accuracy = nltk.classify.util.accuracy(classifier, testfeats)
	five_star_precision = nltk.metrics.precision(refsets['5'], testsets['5'])
	five_star_recall = nltk.metrics.recall(refsets['5'], testsets['5'])
	five_star_fmeasure = nltk.metrics.f_measure(refsets['5'], testsets['5'])
	four_star_precision = nltk.metrics.precision(refsets['4'], testsets['4'])
	four_star_recall = nltk.metrics.recall(refsets['4'], testsets['4'])
	four_star_fmeasure = nltk.metrics.f_measure(refsets['4'], testsets['4'])	
	three_star_precision = nltk.metrics.precision(refsets['3'], testsets['3'])
	three_star_recall = nltk.metrics.recall(refsets['3'], testsets['3'])
	three_star_fmeasure = nltk.metrics.f_measure(refsets['3'], testsets['3'])
	two_star_precision = nltk.metrics.precision(refsets['2'], testsets['2'])
	two_star_recall = nltk.metrics.recall(refsets['2'], testsets['2'])
	two_star_fmeasure = nltk.metrics.f_measure(refsets['2'], testsets['2'])	
	one_star_precision = nltk.metrics.precision(refsets['1'], testsets['1'])
	one_star_recall = nltk.metrics.recall(refsets['1'], testsets['1'])
	one_star_fmeasure =  nltk.metrics.f_measure(refsets['1'], testsets['1'])
	
	print ''
	print '---------------------------------------'
	print 'SINGLE FOLD RESULT ' + '(' + classifierName + ')'
	print '---------------------------------------'
	print 'accuracy:', accuracy
	print 'precision', (five_star_precision + four_star_precision + three_star_precision + two_star_precision + one_star_precision) / 5
	print 'recall', (five_star_recall + four_star_recall + three_star_recall + two_star_recall + one_star_recall) / 5
	print 'f-measure', (five_star_fmeasure + four_star_fmeasure + three_star_fmeasure + two_star_fmeasure + one_star_fmeasure) / 5
				
		#classifier.show_most_informative_features()
	
	print ''
	
	## CROSS VALIDATION
	
	# trainfeats = one_star_feats + three_star_feats + five_star_feats	
	
	# # SHUFFLE TRAIN SET
	# # As in cross validation, the test chunk might have only negative or only positive data	
	# random.shuffle(trainfeats)	
	# n = 5 # 5-fold cross-validation	
			
	# subset_size = len(trainfeats) / n
	# accuracy = []
	# five_star_precision = []
	# five_star_recall = []
	# three_star_precision = []
	# three_star_recall = []
	# one_star_precision = []
	# one_star_recall = []
	# five_star_fmeasure = []
	# three_star_fmeasure = []
	# one_star_fmeasure = []
	# cv_count = 1

	# for i in range(n):		
	# 	testing_this_round = trainfeats[i*subset_size:][:subset_size]
	# 	training_this_round = trainfeats[:i*subset_size] + trainfeats[(i+1)*subset_size:]
		
	# 	classifierName = 'Maximum Entropy'
	# 	classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)

				
	# 	refsets = collections.defaultdict(set)
	# 	testsets = collections.defaultdict(set)
	# 	for i, (feats, label) in enumerate(testing_this_round):
	# 		refsets[label].add(i)
	# 		observed = classifier.classify(feats)
	# 		testsets[observed].add(i)
		
	# 	cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
	# 	cv_five_star_precision = nltk.metrics.precision(refsets['5'], testsets['5'])
	# 	cv_five_star_recall = nltk.metrics.recall(refsets['5'], testsets['5'])
	# 	cv_five_star_fmeasure = nltk.metrics.f_measure(refsets['5'], testsets['5'])
	# 	cv_three_star_precision = nltk.metrics.precision(refsets['3'], testsets['3'])
	# 	cv_three_star_recall = nltk.metrics.recall(refsets['3'], testsets['3'])
	# 	cv_three_star_fmeasure = nltk.metrics.f_measure(refsets['3'], testsets['3'])
	# 	cv_one_star_precision = nltk.metrics.precision(refsets['1'], testsets['1'])
	# 	cv_one_star_recall = nltk.metrics.recall(refsets['1'], testsets['neg'])
	# 	cv_one_star_fmeasure =  nltk.metrics.f_measure(refsets['1'], testsets['1'])
				
	# 	accuracy.append(cv_accuracy)
	# 	five_star_precision.append(cv_five_star_precision)
	# 	five_star_recall.append(cv_five_star_recall)
	# 	three_star_precision.append(cv_three_star_precision)
	# 	three_star_recall.append(cv_three_star_recall)
	# 	one_star_precision.append(cv_one_star_precision)
	# 	one_star_recall.append(cv_one_star_recall)
	# 	five_star_fmeasure.append(cv_five_star_fmeasure)
	# 	three_star_fmeasure.append(cv_three_star_fmeasure)
	# 	one_star_fmeasure.append(cv_one_star_fmeasure)
		
	# 	cv_count += 1
			
	# print '---------------------------------------'
	# print 'N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
	# print '---------------------------------------'
	# print 'accuracy:', sum(accuracy) / n
	# print 'precision', (sum(five_star_precision)/n + sum(three_star_precision)/n + sum(one_star_precision)/n) / 3
	# print 'recall', (sum(five_star_recall)/n + sum(three_star_recall)/n + sum(one_star_recall)/n) / 3
	# print 'f-measure', (sum(five_star_fmeasure)/n + sum(three_star_fmeasure)/n + sum(one_star_fmeasure)/n) / 3
	# print ''
	
		
evaluate_classifier(word_feats)
# evaluate_classifier(stemmed_word_feats)
#evaluate_classifier(stopword_filtered_word_feats)
#evaluate_classifier(bigram_word_feats)	
#evaluate_classifier(bigram_word_feats_stopwords)