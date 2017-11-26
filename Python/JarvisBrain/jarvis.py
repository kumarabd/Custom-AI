import pickle
from nltk.tokenize import word_tokenize

from nltk.classify import ClassifierI
from statistics import mode

abs_path = "/home/hulkbuster/Documents/timepass/JARVIS-master/Python/JarvisBrain/"


class CustomClassifier(ClassifierI):
    def labels(self):
        pass

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)

        return conf

documents_f = open(abs_path+"pickled/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open(abs_path+"pickled/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


open_file = open(abs_path+"pickled/originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/BernoulliNB_classifier.pickle", "rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/SGDC_classifier.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

open_file = open(abs_path+"pickled/NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

custom_classifier = CustomClassifier(
    classifier,
    LinearSVC_classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier,
    SGDC_classifier,
    NuSVC_classifier)


def get_context(text):
    feats = find_features(text)
    return custom_classifier.classify(feats), custom_classifier.confidence(feats)
