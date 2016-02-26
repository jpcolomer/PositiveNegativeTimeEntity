import sys
from pymongo import MongoClient
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
#from nltk.tag import StanfordNERTagger

### UGLY Hack to use StanfordNERTagger
#st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
#st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
#stanford_dir = st._stanford_jar.rpartition('/')[0]
#stanford_jars = find_jars_within_path(stanford_dir)
#st._stanford_jar = ':'.join(stanford_jars)
####

stop_words = set(stopwords.words('english') + ['.', ',', '?', '!', ';', ':'])
include_words = set(['not', 'nor', 'out', 'between', 't', 'against', 'but', 'can', 'no', 'off', 'yes', 'now', 'during'])
stop_words = stop_words - include_words

tagMap = {'NN':wordnet.NOUN,'JJ':wordnet.ADJ,'VB':wordnet.VERB,'RB':wordnet.ADV}
client = MongoClient()
db = client.pnex #PositiveNegativeExample
collection = db.PNCollection # PositiveNegativeCollection

lemmatizer = WordNetLemmatizer()

# Leave 20% for testing purposes

num = int(sys.argv[1])
gt = (num - 1)*0.1
lt = num*0.1
devCursor = db.PNCollection.find({"lemmasDict": { "$exists": False }, "vocabulary_id": { "$exists": False }, "dataPoint.rand": {"$lt": lt, "$gt": gt}})
key = {"vocabulary_id": num}
vocabulary = db.PNCollection.find_one(key)
if vocabulary:
    vocabulary = vocabulary['dictionary']
else:
    vocabulary = {}

def add_negations(sentenceList):
    index = False
    for negation in ["n't", "no", "not"]:
        if negation in sentenceList:
            index = sentenceList.index(negation)
            break
    if index:
        sentenceList = insert_negation(sentenceList, index+1, len(sentenceList))
        sentenceList = insert_negation(sentenceList, 0, index)

    return sentenceList

def insert_negation(sentenceList, start, finish):
    for index in range(start, finish):
        if sentenceList[index] == "TDATE":
            sentenceList[index] = "NOT_TDATE"
        if re.search("[\.,?;:!]", sentenceList[index]) is not None:
            break
    return sentenceList



count = 0
for document in devCursor:
    initialSentence = document['dataPoint']['smearedSentence']
    timeEntities = document['dataPoint']['timeEntityTokens']
    label = document['dataPoint']['label']

    for timeEntity in timeEntities:
        initialSentence = initialSentence.replace(timeEntity, "TDATE")

    sentence = nltk.pos_tag(word_tokenize(initialSentence))

    lemmatizedSentence = []

    for word in sentence:
        if word[1][:2] in tagMap:
            lemmatizedSentence.append(lemmatizer.lemmatize(word[0],tagMap[word[1][:2]]))
        else:
            lemmatizedSentence.append(lemmatizer.lemmatize(word[0]))
    lemmatizedSentence = add_negations(lemmatizedSentence)
    lemmasDict = {}
    for lemma in (set(lemmatizedSentence) - stop_words):
        if re.search("[\.]", lemma) is not None:
            lemma = lemma.replace(".", "\dot")
        if lemma != "TDATE" and lemma != "NOT_TDATE":
            lemma = lemma.lower()
        lemmasDict[lemma] = True
        vocabulary[lemma] = True

    count += 1
    if count % 10 == 0:
        db.PNCollection.update_one(key, {"$set": {"dictionary": vocabulary}}, upsert=True)
    result = db.PNCollection.update_one({'_id': document['_id']}, {"$set": {"lemmasDict": lemmasDict, "POS-TAG": sentence, "lemmatizedSentence": lemmatizedSentence}}, upsert=False)
