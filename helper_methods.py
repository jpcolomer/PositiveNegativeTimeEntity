from nltk.corpus import stopwords
import collections, itertools
import nltk
import re
from nltk import precision
from nltk import recall
from nltk import f_measure

def evaluate_classifier(classifier, validationFeatures):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(validationFeatures):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    accuracy = nltk.classify.util.accuracy(classifier, validationFeatures)
    pos_precision = precision(refsets['POSITIVE_TIME'], testsets['POSITIVE_TIME'])
    pos_recall = recall(refsets['POSITIVE_TIME'], testsets['POSITIVE_TIME'])
    pos_f_measure = f_measure(refsets['POSITIVE_TIME'], testsets['POSITIVE_TIME'])
    neg_precision = precision(refsets['NEGATIVE_TIME'], testsets['NEGATIVE_TIME'])
    neg_recall = recall(refsets['NEGATIVE_TIME'], testsets['NEGATIVE_TIME'])
    neg_f_measure = f_measure(refsets['NEGATIVE_TIME'], testsets['NEGATIVE_TIME'])

    print 'accuracy:', accuracy
    print 'pos precision:', pos_precision
    print 'pos recall:', pos_recall
    print 'pos f-measure', pos_f_measure
    print 'neg precision:', neg_precision
    print 'neg recall:', neg_recall
    print 'neg f-measure', neg_f_measure

    return {'accuracy': accuracy, 'pos precision': pos_precision, 'pos recall': pos_recall, 'pos f-measure': pos_f_measure, 'neg precision': neg_precision, 'neg recall': neg_recall, 'neg f-measure': neg_f_measure}

class PosTagFeatureExtractor:
    def __init__(self):
        self.tags = {}
        self.counter = 0

    def extract(self, tagPairs):
        num = 6
        features = dict([("pos"+str(num), -1) for num in range(0,num)])
        tdate = False
        p = 1
        for tagPair in tagPairs:
            if tagPair[1] not in self.tags:
                self.tags[tagPair[1]] = self.counter
                self.counter += 1
            if tagPair[0] != "TDATE":
                if p > num-1:
                    break
                elif p > num/2-1:
                    features["pos" + str(p)] = self.tags[tagPair[1]]
                else:
                    features["pos0"] = features["pos1"]
                    features["pos1"] = features["pos2"]
                    features["pos2"] = self.tags[tagPair[1]]
            else:
                p += 1

        return features

def remove_stop_words(words):
    stop_words = set(stopwords.words('english') + ['.', ',', '?', '!', ';', ':'])
    include_words = set(['not', 'nor', 'out', 'between', 't', 'against', 'but', 'can', 'no', 'off', 'yes', 'now', 'during'])
    stop_words = stop_words - include_words
    return list(set(words) - stop_words)

def convert_from_postag_to_list(postags):
    return [pair[0] for pair in postags]

def convert_to_dict(sentenceList):
    return dict([(lemma.lower(), True) for lemma in sentenceList])

def add_negations(sentenceList):
    index = False
    # In case the tokenizer didn't do its job
    pattern = "except|never|no|nothing|nowhere|noone|none|not|n't|haven't|hasn't|hadn't|can't|couldn't|shouldn't|won't|wouldn't|don't|doesn't|didn't|isn't|aren't|ain't"
    idx = [i for i, item in enumerate(sentenceList) if re.search(pattern, item)]
    if idx:
        idx = idx[0]
        sentenceList = insert_negation(sentenceList, idx, len(sentenceList))

    return sentenceList

def insert_negation(sentenceList, start, finish):
    for index in range(start, finish):
        if re.search("[\.,?;:!]", sentenceList[index]) is not None:
            break
        elif sentenceList[index] == "NOT_TDATE":
            continue
        else:
            sentenceList[index] = "NOT_" + sentenceList[index]
    return sentenceList
