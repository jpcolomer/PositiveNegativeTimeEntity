from pymongo import MongoClient
from nltk.probability import FreqDist, ConditionalFreqDist
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util, nltk.metrics
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from helper_methods import *
import random
import sklearn
import numpy as np


client = MongoClient()
db = client.pnex


vocaBularyCursor = db.pntime.find({"vocabulary_id": { "$exists": True }})

vocabulary = db.command({'distinct': 'pntime', 'key': 'dictionary'})
vocabulary = vocabulary['values']

dictionary = [set(vocab.keys()) for vocab in vocabulary]
dictionary = set().union(*dictionary)

pos_cursor = db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'POSITIVE_TIME', 'dataPoint.rand': {'$lt': 0.8}})
neg_cursor = db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'NEGATIVE_TIME', 'dataPoint.rand': {'$lt': 0.8}})

pos_tag_extractor = PosTagFeatureExtractor()

def process_documents(cursor):
    features = []
    for document in cursor:
        label = document['dataPoint']['label']
        #feature = convert_to_dict(add_negations(document['lemmatizedSentence']))
        feature = document['lemmatizedSentence']
        #feature = convert_from_postag_to_list(document['POS-TAG'])
        feature = add_negations(feature)
        feature = remove_stop_words(feature)
        feature = convert_to_dict(feature)
        #tag_feature = pos_tag_extractor.extract(document['POS-TAG'])
        #feature.update(tag_feature)
        features.append((feature, label))
    return features

class FeatureSelector:
	def __init__(self, features):
		self.feature_scores = {}
		self.features = features
		self.features_score(features)

	def feature_lists(self, features):
		feature_fd = FreqDist()
		cond_feature_fd = ConditionalFreqDist()
		for feature in features:
			for key in feature[0]:
				feature_fd[key] += 1
				cond_feature_fd[feature[1]][key] += 1
		return (feature_fd, cond_feature_fd)

	def features_score(self, features):
		if self.feature_scores:
			return self.feature_scores
		feature_fd, cond_feature_fd = self.feature_lists(features)
		pos_feature_count = cond_feature_fd['POSITIVE_TIME'].N()
		neg_feature_count = cond_feature_fd['NEGATIVE_TIME'].N()
		total_feature_count = pos_feature_count + neg_feature_count
		self.total_number_of_features = total_feature_count

		for feature, freq in feature_fd.iteritems():
			pos_score = BigramAssocMeasures.chi_sq(cond_feature_fd['POSITIVE_TIME'][feature], (freq, pos_feature_count), total_feature_count)
			neg_score = BigramAssocMeasures.chi_sq(cond_feature_fd['NEGATIVE_TIME'][feature], (freq, neg_feature_count), total_feature_count)
			self.feature_scores[feature] = pos_score + neg_score
		return self.feature_scores

	def best_n_features(self, number):
		featureScores = self.feature_scores
		best_vals = sorted(featureScores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
		best_features = set([w for w, s in best_vals])
		return best_features

	def reprocess_features(self, number):
		best_features = self.best_n_features(number)
		return [(dict([(word, True) for word in feature if word in best_features]), label) for feature, label in self.features]

pos_features = process_documents(pos_cursor)
neg_features = process_documents(neg_cursor)


neg_train_index = int(0.8*len(neg_features))

pos_train_index = int(0.8*len(pos_features))

#nbPosEx = int(pos_train_index*0.05)
#trainFeatures = random.sample(pos_features[pos_train_index:],nbPosEx) + neg_features[neg_train_index:]
validationFeatures = pos_features[:pos_train_index] + neg_features[:neg_train_index]

num = 8.0
#performances = {}
for num in range(1,9):
    num = float(num)
    #performances["cut()".format(num)] = {}
    for i in range(0,int(num)):
        #performances["cut()".format(num)]["cutNb()".format(i)] = {}
        train_features = pos_features[int(i/num*pos_train_index):int((i+1)/num*pos_train_index)] + neg_features[neg_train_index:]
        feature_selector = FeatureSelector(train_features)
        number_of_features = feature_selector.total_number_of_features
        feature_range = [15,50,100,500,1000,2000,5000,10000,15000,number_of_features]
        for feat_num in feature_range:
            #performances["cut()".format(num)]["cutNb()".format(i)]["featsNb()".format(feat_num)] = {}
            train_features = feature_selector.reprocess_features(feat_num)
            print "Evaluating batch ", i, ", ", feat_num,"features and with ", len(train_features), "examples"

            classifier = NaiveBayesClassifier.train(train_features)

            performances = {}
            performances['NB'] = evaluate_classifier(classifier, validationFeatures)

            print "--------------------------"
            print "Linear SVC with L1 penalty"
            LinearSVC_classifier = SklearnClassifier(LinearSVC(penalty='l1', dual=False))
            LinearSVC_classifier.train(train_features)

            performances['SVM L1'] = evaluate_classifier(LinearSVC_classifier, validationFeatures)

            print "--------------------------"
            print "Linear SVC with L2 penalty"
            LinearSVC_classifierL2 = SklearnClassifier(LinearSVC(penalty='l2', dual=True))
            LinearSVC_classifierL2.train(train_features)

            performances['SVM L2'] = evaluate_classifier(LinearSVC_classifierL2, validationFeatures)
            print "--------------------------"
            print "--------------------------"
            value_key = "values.cut({}).cutNb({}).featsNb({})".format(int(num),i,feat_num)
            db.performances.update_one({"performances_id": 1}, {"$set": {value_key: performances}}, upsert=True)

print performances
