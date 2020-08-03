import nltk
from nltk.corpus import treebank
from nltk.tag import DefaultTagger

from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.tag.sequential import ClassifierBasedPOSTagger

data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]
#print(train_data[0])

dt = DefaultTagger('NN')
print(dt.evaluate(test_data))

nt = ClassifierBasedPOSTagger(train=train_data, classifier_builder=NaiveBayesClassifier.train)
print(nt.evaluate(test_data))
