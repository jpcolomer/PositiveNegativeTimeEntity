from pos_tag_feature_extractor import *
from helper_methods import *

class DocumentProcessor:
    def __init__(self, db):
        self.db = db
        self.pos_tag_extractor = PosTagFeatureExtractor()
        self.validationFeatures = False

    def process_documents(self, cursor):
        features = []
        for document in cursor:
            label = document['dataPoint']['label']
            #feature = convert_to_dict(add_negations(document['lemmatizedSentence']))
            feature = document['lemmatizedSentence']
            #feature = convert_from_postag_to_list(document['POS-TAG'])
            feature = add_negations(feature)
            feature = remove_stop_words(feature)
            feature = convert_to_dict(feature)
            #tag_feature = self.pos_tag_extractor.extract(document['POS-TAG'])
            #feature.update(tag_feature)
            features.append((feature, label))
        return features

    def extract_features_from_mongo(self):
        pos_cursor = self.db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'POSITIVE_TIME', 'dataPoint.rand': {'$lt': 0.8}})
        neg_cursor = self.db.pntime.find({"lemmasDict": { "$exists": True }, "vocabulary_id": { "$exists": False }, 'dataPoint.label': 'NEGATIVE_TIME', 'dataPoint.rand': {'$lt': 0.8}})
        self.pos_features = self.process_documents(pos_cursor)
        self.neg_features = self.process_documents(neg_cursor)
        return (self.pos_features, self.neg_features)

    def train_indices(self):
        self.neg_train_index = int(0.8*len(self.neg_features))
        self.pos_train_index = int(0.8*len(self.pos_features))
        return (self.pos_train_index, self.neg_train_index)

    def train_features(self, num,i):
        train_features = self.pos_features[int(i/num*self.pos_train_index):int((i+1)/num*self.pos_train_index)] + self.neg_features[self.neg_train_index:]
        return train_features

    def validation_features(self):
        if self.validationFeatures:
            return self.validationFeatures

        self.train_indices()
        self.validationFeatures = self.pos_features[:self.pos_train_index] + self.neg_features[:self.neg_train_index]
        return self.validationFeatures
