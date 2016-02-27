from pymongo import MongoClient
client = MongoClient()
db = client.pnex


performances = db.performances.find_one({'performances_id': 1})['values']

neg_max_f = {'NB': 0, 'SVM L2': 0, 'SVM L1': 0}
num_max = {}
cutNb_max = {}
featsNb_max = {}

for num in performances:
    for cutNb in performances[num]:
        for featsNb in performances[num][cutNb]:
            for classifier, evaluations in performances[num][cutNb][featsNb].iteritems():
                f = evaluations['neg f-measure']
                if neg_max_f[classifier] < f:
                    neg_max_f[classifier] = f
                    num_max[classifier] = num
                    cutNb_max[classifier] = cutNb
                    featsNb_max[classifier] = featsNb

print neg_max_f
print num_max
print cutNb_max
print featsNb_max
