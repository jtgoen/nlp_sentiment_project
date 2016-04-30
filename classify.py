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
from nltk.stem.wordnet import WordNetLemmatizer

reviews = []

def sanitizeWord(word):
    puncSet = set(['.',',','(',')','?','!',':'])
    for punc in puncSet:
        word = word.replace(punc, "")
    word = word.lower()
    return word

def sanitize(string, remove_stopwords):
	sanitized_words = []
	for word in json.loads(string)['text'].split():
		sanitized_word = sanitizeWord(word)
		if not remove_stopwords or sanitized_word not in stopset:
			sanitized_words.append(sanitized_word)
	return sanitized_words

stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))

def load_data(remove_stopwords):

	reviews[:] = []
	reviews.append([])
	reviews.append([])
	reviews.append([])
	reviews.append([])
	reviews.append([])
	reviews.append([])

	print "Loading data..."
	for line in open('reviews_1_small.json'):
	    reviews[1].append(sanitize(line, remove_stopwords))

	for line in open('reviews_2_small.json'):
	    reviews[2].append(sanitize(line, remove_stopwords))		

	for line in open('reviews_4_small.json'):     	
	    reviews[4].append(sanitize(line, remove_stopwords))		

	for line in open('reviews_5_small.json'):
	    reviews[5].append(sanitize(line, remove_stopwords))
	
def word_feats(words):	
	return dict([(word, True) for word in words])

def stemmed_word_feats(words):
	porter_stemmer = stem.porter.PorterStemmer()
	return dict([(porter_stemmer.stem(word), True) for word in words])

def lemmatized_word_feats(words):
	lemmatizer = stem.wordnet.WordNetLemmatizer()
	return dict([(lemmatizer.lemmatize(word), True) for word in words])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
	if len(words) > 1:
		different_words_exist = False
		first_word = words[0]
		for i in range(1, len(words) - 1):
			if words[i] != first_word:
				different_words_exist = True
				break
		if different_words_exist:
			bigram_finder = BigramCollocationFinder.from_words(words)
			bigrams = bigram_finder.nbest(score_fn, n)
			return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
		else:
			return {}
	else:
		return {}

def multiple_word_feats(words, number_of_features):
	main_dict = dict(word_feats(words))
	if number_of_features > 1:
		feats = stemmed_word_feats(words)
		if bool(feats):
			main_dict.update(feats)
	if number_of_features > 2:
		feats = lemmatized_word_feats(words)
		if bool(feats):
			main_dict.update(feats)
	if number_of_features > 3:
		feats = bigram_word_feats(words)
		if bool(feats):
			main_dict.update(feats)
	return main_dict
 
# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx, number_of_features, remove_stopwords):
	
	print "Adding features..."

	neg_feats = [(featx(f, number_of_features), 'neg') for f in (reviews[1] + reviews[2])]
	pos_feats = [(featx(f, number_of_features), 'pos') for f in (reviews[4] + reviews[5])]

	neg_cutoff = len(neg_feats)*3/4
	pos_cutoff = len(pos_feats)*3/4

	trainfeats = neg_feats[:neg_cutoff] + pos_feats[:pos_cutoff]
	testfeats = neg_feats[neg_cutoff:] + pos_feats[pos_cutoff:]	
 
	classifierName = "Maximum Entropy (Features: Words"
	if remove_stopwords:
		classifierName += ", Removed Stopwords"
	if number_of_features > 1:
		classifierName += ", Stemmed Words"
	if number_of_features > 2:
		classifierName += ", Lemmatized Words"
	if number_of_features > 3:
		classifierName += ", Bigrams"

	classifierName += ")"

	print "Training..."
	classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)
		
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	print "Testing..."
	# for i, (feats, label) in enumerate(testfeats):
	# 	refsets[label].add(i)
	# 	pdist = classifier.prob_classify(feats)
	# 	print 'neg: %f\tpos: %f\t' % (pdist.prob('neg'), pdist.prob('pos'))
	for i, (feats, label) in enumerate(testfeats):
			refsets[label].add(i)
			observed = classifier.classify(feats)
			testsets[observed].add(i)

	accuracy = nltk.classify.util.accuracy(classifier, testfeats)
	
	neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
	neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
	neg_fmeasure =  nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

	pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
	pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
	pos_fmeasure =  nltk.metrics.f_measure(refsets['pos'], testsets['pos'])	

	print ''
	print '---------------------------------------'
	print 'SINGLE FOLD RESULT ' + '(' + classifierName + ')'
	print '---------------------------------------'
	print 'accuracy:', accuracy
	print 'precision', (pos_precision + neg_precision) / 2
	print 'recall', (pos_recall + neg_recall) / 2
	print 'f-measure', (pos_fmeasure + neg_fmeasure) / 2
	print ''
	classifier.show_most_informative_features()
	
	print ''
	
	# CROSS VALIDATION
	
	trainfeats = neg_feats + pos_feats
	# SHUFFLE TRAIN SET
	# As in cross validation, the test chunk might have only negative or only positive data	
	random.shuffle(trainfeats)	
	n = 5 # 5-fold cross-validation	
			
	subset_size = len(trainfeats) / n
	accuracy = []
	five_star_precision = []
	five_star_recall = []
	four_star_precision = []
	four_star_recall = []
	three_star_precision = []
	three_star_recall = []
	two_star_precision = []
	two_star_recall = []
	one_star_precision = []
	one_star_recall = []
	neg_precision = []
	neg_recall = []
	pos_precision = []
	pos_recall = []
	five_star_fmeasure = []
	four_star_fmeasure = []
	three_star_fmeasure = []
	two_star_fmeasure = []
	one_star_fmeasure = []
	neg_fmeasure = []
	pos_fmeasure = []
	cv_count = 1

	for i in range(n):		
		testing_this_round = trainfeats[i*subset_size:][:subset_size]
		training_this_round = trainfeats[:i*subset_size] + trainfeats[(i+1)*subset_size:]
		
		classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)
				
		refsets = collections.defaultdict(set)
		testsets = collections.defaultdict(set)
		for i, (feats, label) in enumerate(testing_this_round):
			refsets[label].add(i)
			observed = classifier.classify(feats)
			testsets[observed].add(i)
		
		cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)

		cv_neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
		cv_neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
		cv_neg_fmeasure =  nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
		cv_pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
		cv_pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
		cv_pos_fmeasure =  nltk.metrics.f_measure(refsets['pos'], testsets['pos'])
				
		accuracy.append(cv_accuracy)

		neg_precision.append(cv_neg_precision)
		neg_recall.append(cv_neg_recall)
		neg_fmeasure.append(cv_neg_fmeasure)
		pos_precision.append(cv_pos_precision)
		pos_recall.append(cv_pos_recall)
		pos_fmeasure.append(cv_pos_fmeasure)
		
		
		cv_count += 1
			
	print '---------------------------------------'
	print 'N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
	print '---------------------------------------'
	print 'accuracy:', sum(accuracy) / n
	print 'precision', (sum(neg_precision)/n + sum(pos_precision)/n) / 2
	print 'recall', (sum(neg_recall)/n + sum(pos_recall)/n) / 2
	print 'f-measure', (sum(neg_fmeasure)/n + sum(pos_fmeasure)/n) / 2	
	print ''
	
		
load_data(True)
evaluate_classifier(multiple_word_feats, 4, True)


