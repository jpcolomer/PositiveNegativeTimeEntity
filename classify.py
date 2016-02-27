from pymongo import MongoClient
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util, nltk.metrics
import random
import sklearn
from feature_selector import *
from document_processor import *
from pos_tag_feature_extractor import *


client = MongoClient()
db = client.pnex

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



def classify_all():
    document_processor = DocumentProcessor(db)
    pos_features, neg_features = document_processor.extract_features_from_mongo()
    pos_train_index, neg_train_index = document_processor.train_indices()


    ## Cross validate for number of features,number of positive buckets and the bucket itself
    num = 8.0
    for num in range(1,9):
        num = float(num)
        for i in range(0,int(num)):
            classify_by_bucket(document_processor,num,i)

def classify_by_bucket(document_processor,num,i):
    train_features = document_processor.train_features(num,i)
    validationFeatures = document_processor.validation_features()
    feature_selector = FeatureSelector(train_features)
    number_of_features = feature_selector.total_number_of_features
    feature_range = [15,50,100,500,1000,2000,5000,10000,15000,number_of_features]
    for feat_num in feature_range:
        multiple_classification(feature_selector, train_features, validationFeatures,num,i,feat_num)

classify_all()
