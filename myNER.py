import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn_crfsuite

df = pd.read_csv('ner_dataset.csv', encoding='ISO-8859-1')
df = df.fillna(method='ffill')

print(df.info())

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {'bias': 1.0,'word.lower()': word.lower(),'word[-3:]': word[-3:],
                'word[-2:]': word[-2:],'word.isupper()': word.isupper(),
                'word.istitle()': word.istitle(),'word.isdigit()': word.isdigit(),
    'postag': postag,'postag[:2]': postag[:2],}

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features
# convert input sentence into features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
# get corresponding outcome NER tag label for input sentence
def sent2labels(sent):
    return [label for token, postag, label in sent]

agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),s['POS'].values.tolist(),s['Tag'].values.tolist())]
grouped_df = df.groupby('Sentence #').apply(agg_func)

sentences = [s for s in grouped_df]

X = np.array([sent2features(s) for s in sentences])
y = np.array([sent2labels(s) for s in sentences])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

crf = sklearn_crfsuite.CRF(algorithm='lbfgs',
                           c1=0.1,
                           c2=0.1,
                           max_iterations=100,
                           all_possible_transitions=True,
                           verbose=True)
crf.fit(X_train, y_train)