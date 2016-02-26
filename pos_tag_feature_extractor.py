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
