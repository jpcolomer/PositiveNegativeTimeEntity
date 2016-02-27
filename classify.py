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
from collections import Counter


client = MongoClient()
db = client.pnex

def multiple_classification(train_features, validationFeatures,num,i,feat_num):
    print "Evaluating batch ", i, ", ", feat_num,"features and with ", len(train_features), "examples"

    classifier = NaiveBayesClassifier.train(train_features)

    performances = {}
    performances['NB'] = evaluate_classifier(classifier, validationFeatures)

    print "--------------------------"
    print "Linear SVC with L1 penalty"
    LinearSVC_classifier = SklearnClassifier(LinearSVC(penalty='l1', dual=False))
    LinearSVC_classifier.train(train_features)

    performances['SVM L2'] = evaluate_classifier(LinearSVC_classifier, validationFeatures)

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
        train_features = feature_selector.reprocess_features(feat_num)
        multiple_classification(train_features, validationFeatures,num,i,feat_num)

def classify_one_combination(num,i,feat_num=None):
    classification_parameters= get_parameters_to_classify(num,i,feat_num)
    train_features = classification_parameters[0]
    validationFeatures = classification_parameters[1]
    multiple_classification(train_features, validationFeatures,num,i,feat_num)

def get_parameters_to_classify(num,i,feat_num=None):
    document_processor = DocumentProcessor(db)
    pos_features, neg_features = document_processor.extract_features_from_mongo()
    pos_train_index, neg_train_index = document_processor.train_indices()
    train_features = document_processor.train_features(num,i)
    validationFeatures = document_processor.validation_features()
    if feat_num:
        feature_selector = FeatureSelector(train_features)
        number_of_features = feature_selector.total_number_of_features
        train_features = feature_selector.reprocess_features(feat_num)
    train_features = [(Counter(features)+Counter(document_processor.tag_features[i]), label) for i, (features, label) in enumerate(train_features)]
    return (train_features, validationFeatures)


def evaluate_on_test_data():
    # The best model is SVM L1 on validation data. The best combination of features is
    # using lemmatized sentence with the pos tag of the neighboors of the Time Entity
    # The Positive examples were separated into "num" divisions with same number of examples
    # Each division was used as train data and evaluated with the validation data

    document_processor = DocumentProcessor(db)
    pos_features, neg_features = document_processor.extract_features_from_mongo()
    pos_train_index, neg_train_index = document_processor.train_indices()

    # Get train features for number of divisions 6 and division 2
    num = 6.0
    i = 2
    feat_num = 48013
    train_features = document_processor.train_features(num,i)
    #######

    # Usually, after cross validation I would train again with the entire dev data set (train + validation)
    # and then see the performance of the final model with the test dataset, but since the entire dataset is so
    # inbalanced I'm just using the rebalanced train data set as the final model.

    feature_selector = FeatureSelector(train_features)
    number_of_features = feature_selector.total_number_of_features
    train_features = feature_selector.reprocess_features(feat_num)
    # Add POS tag features
    train_features = [(Counter(features)+Counter(document_processor.tag_features[i]), label) for i, (features, label) in enumerate(train_features)]
    #
    ### Get testing data. This wasn't touched during development
    testing_data = document_processor.testing_data()

    print "--------------------------"
    print "Linear SVC with L1 penalty"
    LinearSVC_classifier = SklearnClassifier(LinearSVC(penalty='l1', dual=False, C=0.7))
    LinearSVC_classifier.train(train_features)

    evaluate_classifier(LinearSVC_classifier, testing_data)


#classify_one_combination(6.0,2,48013)
evaluate_on_test_data()
