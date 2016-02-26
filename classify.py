from pymongo import MongoClient
import collections, itertools
import nltk
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util, nltk.metrics
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import precision
from nltk import recall
from helper_methods import *
import sklearn


client = MongoClient()
db = client.pnex


vocaBularyCursor = db.pntime.find({"vocabulary_id": { "$exists": True }})

vocabulary = db.command({'distinct': 'pntime', 'key': 'dictionary'})
vocabulary = vocabulary['values']

dictionary = [set(vocab.keys()) for vocab in vocabulary]
dictionary = set().union(*dictionary)

posCursor = db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'POSITIVE_TIME', 'dataPoint.rand': {'$lt': 0.05}})
negCursor = db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'NEGATIVE_TIME', 'dataPoint.rand': {'$lt': 0.6}})


features = []

def processDocuments(cursor):
    for document in cursor:
        label = document['dataPoint']['label']
        #feature = convert_to_dict(add_negations(document['lemmatizedSentence']))
        feature = document['lemmatizedSentence']
        #feature = convert_from_postag_to_list(document['POS-TAG'])
        #feature = add_negations(feature)
        feature = remove_stop_words(feature)
        feature = convert_to_dict(feature)
        features.append((feature, label))

processDocuments(posCursor)
processDocuments(negCursor)

trainIndex = int(0.9*len(features))
trainFeatures = features[trainIndex:]
validationFeatures = features[:trainIndex]
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

classifier = NaiveBayesClassifier.train(features)
for i, (feats, label) in enumerate(validationFeatures):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

print 'accuracy:', nltk.classify.util.accuracy(classifier, validationFeatures)
print 'pos precision:', precision(refsets['POSITIVE_TIME'], testsets['POSITIVE_TIME'])
print 'pos recall:', recall(refsets['POSITIVE_TIME'], testsets['POSITIVE_TIME'])
print 'neg precision:', precision(refsets['NEGATIVE_TIME'], testsets['NEGATIVE_TIME'])
print 'neg recall:', recall(refsets['NEGATIVE_TIME'], testsets['NEGATIVE_TIME'])


