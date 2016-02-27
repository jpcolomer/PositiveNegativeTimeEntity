from pymongo import MongoClient
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from classify import *

client = MongoClient()
db = client.pnex


def cross_validate_L1_C(db):
    num = 6.0
    i = 2
    feat_num = 48013
    classification_parameters= get_parameters_to_classify(num,i,feat_num)
    train_features = classification_parameters[0]
    validationFeatures = classification_parameters[1]

    neg_f = []
    pos_f = []
    c_range = list(np.linspace(0.1,2,20)) + [5, 10, 25, 50, 100, 250, 500, 750, 1000, 2500, 5000]
    for C in c_range:
        LinearSVC_classifier = SklearnClassifier(LinearSVC(penalty='l1', dual=False, C=C))
        LinearSVC_classifier.train(train_features)

        performance = evaluate_classifier(LinearSVC_classifier, validationFeatures)
        neg_f.append(performance['neg f-measure'])
        pos_f.append(performance['pos f-measure'])

    db.performances.update_one({"performances_id": 'opt-model'}, {"$set": {'c_range': c_range, 'neg_f': neg_f, 'pos_f': pos_f}}, upsert=True)




def get_max_measure(measure):
    performances = db.performances.find_one({'performances_id': 1})['values']
    max_measure = {'NB': 0, 'SVM L2': 0, 'SVM L1': 0}
    num_max = {}
    cutNb_max = {}
    featsNb_max = {}

    max_measure_per_featsNb = {}
    for num in performances:
        for cutNb in performances[num]:
            for featsNb in performances[num][cutNb]:
                for classifier, evaluations in performances[num][cutNb][featsNb].iteritems():
                    if classifier not in max_measure_per_featsNb:
                        max_measure_per_featsNb[classifier] = {}
                    if featsNb not in max_measure_per_featsNb[classifier]:
                        max_measure_per_featsNb[classifier][featsNb] = 0
                    f = evaluations[measure]
                    if max_measure[classifier] < f:
                        max_measure[classifier] = f
                        num_max[classifier] = num
                        cutNb_max[classifier] = cutNb
                        featsNb_max[classifier] = featsNb

                    if max_measure_per_featsNb[classifier][featsNb] < f:
                        max_measure_per_featsNb[classifier][featsNb] = f
    print max_measure
    print num_max
    print cutNb_max
    print featsNb_max
    return (max_measure_per_featsNb, max_measure, num_max, cutNb_max, featsNb_max)


def plot_max_measure_per_featsNb(measure_str):
    max_measure = get_max_measure(measure_str)[0]
    max_measure_array = {}
    classifiers = ['SVM L1', 'SVM L2', 'NB']
    feats = []
    measure = []
    for classifier in classifiers:
        max_measure_array[classifier] = {}
        maximum = 0
        for key, value in max_measure[classifier].iteritems():
            number = int(re.findall(r'\d+', key)[0])
            if number in [15,50,100,500,1000,2000,5000,10000,15000]:
                max_measure_array[classifier][number] = value
            if maximum < value:
                maximum = value
                maximum_num = number

        max_measure_array[classifier][maximum_num] = maximum

        feats.append([key for key, value in sorted(max_measure_array[classifier].items())])
        measure.append([value for key, value in sorted(max_measure_array[classifier].items())])
    f = plt.figure()
    line1 = plt.plot(feats[0], measure[0],'r', label=classifiers[0])
    line2 = plt.plot(feats[1], measure[1],'b', label=classifiers[1])
    line3 = plt.plot(feats[2], measure[2],'g', label=classifiers[2])
    plt.title(measure_str)
    plt.xlabel('Number of features')
    plt.legend(loc=2, borderaxespad=0.)
    plt.xscale('log')
    f.savefig('{0}.png'.format(measure_str))
    plt.close(f)

def plot_c_cross_validation(db):
    performances = db.performances.find_one({"performances_id": 'opt-model'})
    neg_f = performances['neg_f']
    pos_f = performances['pos_f']
    c_range = performances['c_range']
    max_neg_f = max(neg_f)
    index_max = c_range[neg_f.index(max_neg_f)]
    f = plt.figure()
    line1 = plt.plot(c_range, neg_f,'r', label="Negative f measure")
    line2 = plt.plot(c_range, pos_f,'b', label="Positive f measure")
    line3 = plt.plot(index_max, max_neg_f,'o', label="Max Neg f = {:.3f}".format(max_neg_f))
    plt.title("Cross validation for cost C in L1")
    plt.xlabel('C')
    plt.legend(loc=2, borderaxespad=0.)
    plt.xscale('log')
    f.savefig('{0}.png'.format('c_cross_val'))
    plt.close(f)



#plot_max_measure_per_featsNb('neg f-measure')
#plot_max_measure_per_featsNb('pos f-measure')
#cross_validate_L1_C(db)
plot_c_cross_validation(db)
