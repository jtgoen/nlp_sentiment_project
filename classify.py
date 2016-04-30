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
import nltk.data

reviews = []
reviews_sents = []
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def tokenize_sentences(review):
    sentences = sent_detector.tokenize(review.strip())
    return sentences


def sanitizeWord(word):
    puncSet = set(['.', ',', '(', ')', '?', '!', ':'])
    for punc in puncSet:
        word = word.replace(punc, "")
    word = word.lower()
    return word


def sanitize(words, remove_stopwords):
    sanitized_words = []
    for word in words.split():
        sanitized_word = sanitizeWord(word)
        if not remove_stopwords or sanitized_word not in stopset:
            sanitized_words.append(sanitized_word)
    return sanitized_words


stopset = set(stopwords.words('english')) - set(('over', 'under', 'below', 'more', 'most', 'no', 'not', 'only', 'such',
                                                 'few', 'so', 'too', 'very', 'just', 'any', 'once'))


def load_data(remove_stopwords):
    reviews[:] = []
    reviews.append([])
    reviews.append([])
    reviews.append([])
    reviews.append([])
    reviews.append([])
    reviews.append([])

    reviews_sents[:] = []
    reviews_sents.append([])
    reviews_sents.append([])
    reviews_sents.append([])
    reviews_sents.append([])
    reviews_sents.append([])
    reviews_sents.append([])

    print "Loading data..."
    for line in open('reviews_1_small.json'):
        reviews[1].append(sanitize(json.loads(line)['text'], remove_stopwords))
        sentences = tokenize_sentences(json.loads(line)['text'])
        sentence_bags = []
        for sentence in sentences:
            sentence_bags.append(sanitize(sentence, remove_stopwords))
        reviews_sents[1].append(sentence_bags)

    for line in open('reviews_2_small.json'):
        reviews[2].append(sanitize(json.loads(line)['text'], remove_stopwords))
        sentences = tokenize_sentences(json.loads(line)['text'])
        sentence_bags = []
        for sentence in sentences:
            sentence_bags.append(sanitize(sentence, remove_stopwords))
        reviews_sents[2].append(sentence_bags)

    for line in open('reviews_4_small.json'):
        reviews[4].append(sanitize(json.loads(line)['text'], remove_stopwords))
        sentences = tokenize_sentences(json.loads(line)['text'])
        sentence_bags = []
        for sentence in sentences:
            sentence_bags.append(sanitize(sentence, remove_stopwords))
        reviews_sents[4].append(sentence_bags)

    for line in open('reviews_5_small.json'):
        reviews[5].append(sanitize(json.loads(line)['text'], remove_stopwords))
        sentences = tokenize_sentences(json.loads(line)['text'])
        sentence_bags = []
        for sentence in sentences:
            sentence_bags.append(sanitize(sentence, remove_stopwords))
        reviews_sents[5].append(sentence_bags)


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

    # create labeled bags of words (dictionary)
    neg_reviews = reviews[1] + reviews[2]
    pos_reviews = reviews[4] + reviews[5]
    random.shuffle(neg_reviews)
    random.shuffle(pos_reviews)
    neg_feats = [(featx(f, number_of_features), 'neg') for f in neg_reviews]
    pos_feats = [(featx(f, number_of_features), 'pos') for f in pos_reviews]

    neg_cutoff = len(neg_feats) * 3 / 4
    pos_cutoff = len(pos_feats) * 3 / 4

    trainfeats = neg_feats[:neg_cutoff] + pos_feats[:pos_cutoff]
    testfeats = neg_feats[neg_cutoff:] + pos_feats[pos_cutoff:]

    neg_sent_reviews = reviews_sents[1] + reviews_sents[2]
    pos_sent_reviews = reviews_sents[4] + reviews_sents[5]
    random.shuffle(neg_sent_reviews)
    random.shuffle(pos_sent_reviews)
    neg_sent_feats = [(sentences, 'neg') for sentences in neg_sent_reviews]
    pos_sent_feats = [(sentences, 'pos') for sentences in pos_sent_reviews]
    test_sent_feats = neg_sent_feats[neg_cutoff:] + pos_sent_feats[pos_cutoff:]

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
    classifier = MaxentClassifier.train(trainfeats, 'GIS', trace=0, encoding=None, labels=None, sparse=True,
                                        gaussian_prior_sigma=0, max_iter=1)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    sent_refsets = collections.defaultdict(set)
    sent_testsets = collections.defaultdict(set)

    print "Testing..."
    # for i, (feats, label) in enumerate(testfeats):
    #   refsets[label].add(i)
    #   pdist = classifier.prob_classify(feats)
    #   print 'neg: %f\tpos: %f\t' % (pdist.prob('neg'), pdist.prob('pos'))
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print "Testing on Sentences..."

    for i, (sentences, label) in enumerate(test_sent_feats):
        sent_refsets[label].add(i)
        pos_prob_total = 0
        neg_prob_total = 0
        sentence_count = len(sentences)
        for sentence in sentences:
            pdist = classifier.prob_classify(featx(sentence, 4))
            pos_prob_total += pdist.prob('pos')
            neg_prob_total += pdist.prob('neg')
        if ((pos_prob_total / sentence_count) > (neg_prob_total / sentence_count)):
            results = 'pos'
        else:
            results = 'neg'
        sent_testsets[results].add(i)
    accuracy = nltk.classify.util.accuracy(classifier, testfeats)
    #accuracy = nltk.classify.util.accuracy(classifier, test_sent_feats)

    neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
    neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
    neg_fmeasure =  nltk.metrics.f_measure(refsets['neg'], testsets['neg'])

    pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
    pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
    pos_fmeasure =  nltk.metrics.f_measure(refsets['pos'], testsets['pos'])

    sent_neg_precision = nltk.metrics.precision(sent_refsets['neg'], sent_testsets['neg'])
    sent_neg_recall = nltk.metrics.recall(sent_refsets['neg'], sent_testsets['neg'])
    sent_neg_fmeasure = nltk.metrics.f_measure(sent_refsets['neg'], sent_testsets['neg'])

    sent_pos_precision = nltk.metrics.precision(sent_refsets['pos'], sent_testsets['pos'])
    sent_pos_recall = nltk.metrics.recall(sent_refsets['pos'], sent_testsets['pos'])
    sent_pos_fmeasure = nltk.metrics.f_measure(sent_refsets['pos'], sent_testsets['pos'])

    print ''
    print '---------------------------------------'
    print 'SINGLE FOLD RESULT ' + '(' + classifierName + ')'
    print '---------------------------------------'
    print 'accuracy:', accuracy
    print 'precision', (pos_precision + neg_precision) / 2
    print 'recall', (pos_recall + neg_recall) / 2
    print 'f-measure', (pos_fmeasure + neg_fmeasure) / 2
    print ''

    print ''
    print '---------------------------------------'
    print 'SENTENCES: SINGLE FOLD RESULT ' + '(' + classifierName + ')'
    print '---------------------------------------'
    # print 'accuracy:', accuracy
    print 'precision', (sent_pos_precision + sent_neg_precision) / 2
    print 'recall', (sent_pos_recall + sent_neg_recall) / 2
    print 'f-measure', (sent_pos_fmeasure + sent_neg_fmeasure) / 2
    print ''

    # CROSS VALIDATION

    trainfeats = neg_feats + pos_feats
    test_sent_feats = neg_sent_feats + pos_sent_feats
    # SHUFFLE TRAIN SET
    # As in cross validation, the test chunk might have only negative or only positive data
    random.shuffle(trainfeats)
    n = 5  # 5-fold cross-validation

    subset_size = len(trainfeats) / n
    accuracy = []
    neg_precision = []
    neg_recall = []
    pos_precision = []
    pos_recall = []
    neg_fmeasure = []
    pos_fmeasure = []
    sent_neg_precision = []
    sent_neg_recall = []
    sent_pos_precision = []
    sent_pos_recall = []
    sent_neg_fmeasure = []
    sent_pos_fmeasure = []    
    cv_count = 1

    print 'Starting 5-fold cross validation...'
    for i in range(n):
        print "Fold " + str(i) + ":"
        testing_this_round = trainfeats[i * subset_size:][:subset_size]
        sent_testing_this_round = test_sent_feats[i * subset_size:][:subset_size]
        training_this_round = trainfeats[:i * subset_size] + trainfeats[(i + 1) * subset_size:]

        classifier = MaxentClassifier.train(training_this_round, 'GIS', trace=0, encoding=None, labels=None,
                                            sparse=True, gaussian_prior_sigma=0, max_iter=1)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        sent_refsets = collections.defaultdict(set)
        sent_testsets = collections.defaultdict(set)

        print "Testing..."

        for i, (feats, label) in enumerate(testing_this_round):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)

        print "Testing on Sentences..."

        for i, (sentences, label) in enumerate(sent_testing_this_round):
            sent_refsets[label].add(i)
            pos_prob_total = 0
            neg_prob_total = 0
            sentence_count = len(sentences)
            for sentence in sentences:
                pdist = classifier.prob_classify(featx(sentence, 4))
                pos_prob_total += pdist.prob('pos')
                neg_prob_total += pdist.prob('neg')
            if ((pos_prob_total / sentence_count) > (neg_prob_total / sentence_count)):
                results = 'pos'
            else:
                results = 'neg'
            sent_testsets[results].add(i)            

        cv_accuracy = nltk.classify.util.accuracy(classifier, testing_this_round)

        cv_neg_precision = nltk.metrics.precision(refsets['neg'], testsets['neg'])
        cv_neg_recall = nltk.metrics.recall(refsets['neg'], testsets['neg'])
        cv_neg_fmeasure = nltk.metrics.f_measure(refsets['neg'], testsets['neg'])
        cv_pos_precision = nltk.metrics.precision(refsets['pos'], testsets['pos'])
        cv_pos_recall = nltk.metrics.recall(refsets['pos'], testsets['pos'])
        cv_pos_fmeasure = nltk.metrics.f_measure(refsets['pos'], testsets['pos'])

        sent_cv_neg_precision = nltk.metrics.precision(sent_refsets['neg'], sent_testsets['neg'])
        sent_cv_neg_recall = nltk.metrics.recall(sent_refsets['neg'], sent_testsets['neg'])
        sent_cv_neg_fmeasure = nltk.metrics.f_measure(sent_refsets['neg'], sent_testsets['neg'])
        sent_cv_pos_precision = nltk.metrics.precision(sent_refsets['pos'], sent_testsets['pos'])
        sent_cv_pos_recall = nltk.metrics.recall(sent_refsets['pos'], sent_testsets['pos'])
        sent_cv_pos_fmeasure = nltk.metrics.f_measure(sent_refsets['pos'], sent_testsets['pos'])

        accuracy.append(cv_accuracy)

        neg_precision.append(cv_neg_precision)
        neg_recall.append(cv_neg_recall)
        neg_fmeasure.append(cv_neg_fmeasure)
        pos_precision.append(cv_pos_precision)
        pos_recall.append(cv_pos_recall)
        pos_fmeasure.append(cv_pos_fmeasure)

        sent_neg_precision.append(sent_cv_neg_precision)
        sent_neg_recall.append(sent_cv_neg_recall)
        sent_neg_fmeasure.append(sent_cv_neg_fmeasure)
        sent_pos_precision.append(sent_cv_pos_precision)
        sent_pos_recall.append(sent_cv_pos_recall)
        sent_pos_fmeasure.append(sent_cv_pos_fmeasure)

        cv_count += 1

    print '---------------------------------------'
    print 'N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
    print '---------------------------------------'
    print 'accuracy:', sum(accuracy) / n
    print 'precision', (sum(neg_precision) / n + sum(pos_precision) / n) / 2
    print 'recall', (sum(neg_recall) / n + sum(pos_recall) / n) / 2
    print 'f-measure', (sum(neg_fmeasure) / n + sum(pos_fmeasure) / n) / 2
    print ''

    print '---------------------------------------'
    print 'SENTENCES: N-FOLD CROSS VALIDATION RESULT ' + '(' + classifierName + ')'
    print '---------------------------------------'
    # print 'accuracy:', sum(accuracy) / n
    print 'precision', (sum(sent_neg_precision) / n + sum(sent_pos_precision) / n) / 2
    print 'recall', (sum(sent_neg_recall) / n + sum(sent_pos_recall) / n) / 2
    print 'f-measure', (sum(sent_neg_fmeasure) / n + sum(sent_pos_fmeasure) / n) / 2
    print ''


load_data(True)
evaluate_classifier(multiple_word_feats, 4, True)
