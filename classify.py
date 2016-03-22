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

def sanitizeWord(word):
    puncSet = set(['.',',','(',')','?','!',':'])
    for punc in puncSet:
        word = word.replace(punc, "")
    word = word.lower()
    return word

def sanitize(string):
	sanitized_words = []
	for word in json.loads(string)['text'].split():
		sanitized_word = sanitizeWord(word)
		if sanitized_word not in stopset:
			sanitized_words.append(sanitized_word)
	return sanitized_words

stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such', 'few', 'so', 'too', 'very', 'just', 'any', 'once'))
reviews_1 = []
reviews_2 = []
reviews_3 = []
reviews_4 = []
reviews_5 = []

def load_data():
	print "Loading data..."
	for line in open('reviews_1_small.json'):
	    reviews_1.append(sanitize(line))

	for line in open('reviews_2_small.json'):
	    reviews_2.append(sanitize(line))		

	for line in open('reviews_3_small.json'):
	    reviews_3.append(sanitize(line))		

	for line in open('reviews_4_small.json'):     	
	    reviews_4.append(sanitize(line))		

	for line in open('reviews_5_small.json'):
	    reviews_5.append(sanitize(line))
	
def word_feats(words):	
	return dict([(word, True) for word in words])

def stemmed_word_feats(words):
	porter_stemmer = stem.porter.PorterStemmer()
	return dict([(porter_stemmer.stem(word), True) for word in words])

def lemmatized_word_feats(words):
	lemmatizer = stem.wordnet.WordNetLemmatizer()
	return dict([(lemmatizer.lemmatize(word), True) for word in words])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
	bigram_finder = BigramCollocationFinder.from_words(words)
	bigrams = bigram_finder.nbest(score_fn, n)
	return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])

def multiple_word_feats(words, number_of_features):
	main_dict = dict(word_feats(words))
	main_dict.update(stemmed_word_feats(words))
	main_dict.update(lemmatized_word_feats(words))
	main_dict.update(bigram_word_feats(words))
	return main_dict
 
# Calculating Precision, Recall & F-measure
def evaluate_classifier(featx, back_half):
	
	print "Adding features..."
	if back_half:
		back_reviews_1 = reviews_1[(len(reviews_1)*1/2):]
		back_reviews_2 = reviews_2[(len(reviews_1)*1/2):]
		back_reviews_3 = reviews_3[(len(reviews_1)*1/2):]
		back_reviews_4 = reviews_4[(len(reviews_1)*1/2):]
		back_reviews_5 = reviews_5[(len(reviews_1)*1/2):]

		one_star_feats = [(featx(f), '1') for f in back_reviews_1]
		two_star_feats = [(featx(f), '2') for f in back_reviews_2]
		three_star_feats = [(featx(f), '3') for f in back_reviews_3]
		four_star_feats = [(featx(f), '4') for f in back_reviews_4]
		five_star_feats = [(featx(f), '5') for f in back_reviews_5]
	else:
		one_star_feats = [(featx(f), '1') for f in reviews_1]
		two_star_feats = [(featx(f), '2') for f in reviews_2]
		three_star_feats = [(featx(f), '3') for f in reviews_3]
		four_star_feats = [(featx(f), '4') for f in reviews_4]
		five_star_feats = [(featx(f), '5') for f in reviews_5]

	    
	one_star_cutoff = len(one_star_feats)*3/4
	two_star_cutoff = len(two_star_feats)*3/4
	three_star_cutoff = len(three_star_feats)*3/4
	four_star_cutoff = len(four_star_feats)*3/4
	five_star_cutoff = len(five_star_feats)*3/4
 
	trainfeats = one_star_feats[:one_star_cutoff] + two_star_feats[:two_star_cutoff] + three_star_feats[:three_star_cutoff] + four_star_feats[:four_star_cutoff] + five_star_feats[:five_star_cutoff]
	testfeats = one_star_feats[one_star_cutoff:] + two_star_feats[two_star_cutoff:] + three_star_feats[three_star_cutoff:] + four_star_feats[four_star_cutoff:] + five_star_feats[five_star_cutoff:]


	if back_half:
		classifierName = 'Maximum Entropy'
	else:
		classifierName = 'Maximum Entropy'

	print "Training..."
	classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)

		
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	print "Testing..."
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
	print ''
	classifier.show_most_informative_features()
	
	print ''
	
	# CROSS VALIDATION
	
	trainfeats = one_star_feats + two_star_feats + three_star_feats + four_star_feats + five_star_feats
	
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
	five_star_fmeasure = []
	four_star_fmeasure = []
	three_star_fmeasure = []
	two_star_fmeasure = []
	one_star_fmeasure = []
	cv_count = 1

	for i in range(n):		
		testing_this_round = trainfeats[i*subset_size:][:subset_size]
		training_this_round = trainfeats[:i*subset_size] + trainfeats[(i+1)*subset_size:]
		
		classifierName = 'Maximum Entropy'
		classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 1)
				
		refsets = collections.defaultdict(set)
		testsets = collections.defaultdict(set)
		for i, (feats, label) in enumerate(testing_this_round):
			refsets[label].add(i)
			observed = classifier.classify(feats)
			testsets[observed].add(i)
		
		cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)
		cv_five_star_precision = nltk.metrics.precision(refsets['5'], testsets['5'])
		cv_five_star_recall = nltk.metrics.recall(refsets['5'], testsets['5'])
		cv_five_star_fmeasure = nltk.metrics.f_measure(refsets['5'], testsets['5'])
		cv_four_star_precision = nltk.metrics.precision(refsets['4'], testsets['4'])
		cv_four_star_recall = nltk.metrics.recall(refsets['4'], testsets['4'])
		cv_four_star_fmeasure = nltk.metrics.f_measure(refsets['4'], testsets['4'])
		cv_three_star_precision = nltk.metrics.precision(refsets['3'], testsets['3'])
		cv_three_star_recall = nltk.metrics.recall(refsets['3'], testsets['3'])
		cv_three_star_fmeasure = nltk.metrics.f_measure(refsets['3'], testsets['3'])
		cv_two_star_precision = nltk.metrics.precision(refsets['2'], testsets['2'])
		cv_two_star_recall = nltk.metrics.recall(refsets['2'], testsets['2'])
		cv_two_star_fmeasure = nltk.metrics.f_measure(refsets['2'], testsets['2'])
		cv_one_star_precision = nltk.metrics.precision(refsets['1'], testsets['1'])
		cv_one_star_recall = nltk.metrics.recall(refsets['1'], testsets['neg'])
		cv_one_star_fmeasure =  nltk.metrics.f_measure(refsets['1'], testsets['1'])
				
		accuracy.append(cv_accuracy)
		five_star_precision.append(cv_five_star_precision)
		five_star_recall.append(cv_five_star_recall)
		four_star_precision.append(cv_four_star_precision)
		four_star_recall.append(cv_four_star_recall)
		three_star_precision.append(cv_three_star_precision)
		three_star_recall.append(cv_three_star_recall)
		two_star_precision.append(cv_two_star_precision)
		two_star_recall.append(cv_two_star_recall)
		one_star_precision.append(cv_one_star_precision)
		one_star_recall.append(cv_one_star_recall)
		five_star_fmeasure.append(cv_five_star_fmeasure)
		four_star_fmeasure.append(cv_four_star_fmeasure)
		three_star_fmeasure.append(cv_three_star_fmeasure)
		two_star_fmeasure.append(cv_two_star_fmeasure)
		one_star_fmeasure.append(cv_one_star_fmeasure)
		
		cv_count += 1
			
	print '---------------------------------------'
	print 'N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
	print '---------------------------------------'
	print 'accuracy:', sum(accuracy) / n
	print 'precision', (sum(five_star_precision)/n + sum(four_star_precision)/n + sum(three_star_precision)/n + sum(two_star_precision)/n + sum(one_star_precision)/n) / 5
	print 'recall', (sum(five_star_recall)/n + sum(four_star_recall)/n + sum(three_star_recall)/n + sum(two_star_recall)/n + sum(one_star_recall)/n) / 5
	print 'f-measure', (sum(five_star_fmeasure)/n + sum(four_star_fmeasure)/n + sum(three_star_fmeasure)/n + sum(two_star_fmeasure)/n + sum(one_star_fmeasure)/n) / 5
	print ''
	
		
# evaluate_classifier(word_feats)
# evaluate_classifier(stemmed_word_feats)
# evaluate_classifier(multiple_word_feats, False)
load_data()
evaluate_classifier(multiple_word_feats, True)
#evaluate_classifier(stopword_filtered_word_feats)
#evaluate_classifier(bigram_word_feats)	
#evaluate_classifier(bigram_word_feats_stopwords)