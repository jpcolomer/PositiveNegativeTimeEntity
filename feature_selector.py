from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures

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
