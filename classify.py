from pymongo import MongoClient
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util, nltk.metrics
from helper_methods import *
import random
import sklearn
from feature_selector import *
from pos_tag_feature_extractor import *


client = MongoClient()
db = client.pnex

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


def multiple_classification(feature_selector, train_featurs, validationFeatures,num,i,feat_num):
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
    #db.performances.update_one({"performances_id": 1}, {"$set": {value_key: performances}}, upsert=True)

pos_features = process_documents(pos_cursor)
neg_features = process_documents(neg_cursor)


neg_train_index = int(0.8*len(neg_features))
pos_train_index = int(0.8*len(pos_features))

#nbPosEx = int(pos_train_index*0.05)
#trainFeatures = random.sample(pos_features[pos_train_index:],nbPosEx) + neg_features[neg_train_index:]
validationFeatures = pos_features[:pos_train_index] + neg_features[:neg_train_index]

## Cross validate for number of features,number of positive buckets and the bucket itself
num = 8.0
for num in range(1,9):
    num = float(num)
    for i in range(0,int(num)):
        train_features = pos_features[int(i/num*pos_train_index):int((i+1)/num*pos_train_index)] + neg_features[neg_train_index:]
        feature_selector = FeatureSelector(train_features)
        number_of_features = feature_selector.total_number_of_features
        feature_range = [15,50,100,500,1000,2000,5000,10000,15000,number_of_features]
        for feat_num in feature_range:
            multiple_classification(feature_selector, train_features, validationFeatures,num,i,feat_num)
