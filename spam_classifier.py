import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from sklearn import model_selection, preprocessing, metrics, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en")

data = pd.read_csv("input/spam.csv", encoding ='latin1')
data = data[['v1', 'v2']]
data = data.rename(columns={'v1':'target', 'v2':'email'})

for i, text in enumerate(data["email"]):
    data.loc[i,"email"] = data.loc[i,"email"].lower()

for i, text in enumerate(data["email"]):
    data.loc[i,"email"] = ' '.join([x.text for x in nlp(data.loc[i,"email"]) if x.text  not in nlp.Defaults.stop_words])

st = PorterStemmer()
for i, text in enumerate(data["email"]):
    data.loc[i,"email"] = ' '.join([st.stem(x.text) for x in nlp(data.loc[i,"email"])])

for i, text in enumerate(data["email"]):
    data.loc[i,"email"] = ' '.join([x.lemma_ for x in nlp(data.loc[i,"email"])])


X_train, X_test, y_train, y_test = model_selection.train_test_split(data["email"], data["target"])
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

tfidf = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf.fit(X_train)

X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)

model = naive_bayes.MultinomialNB(alpha=0.5)
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("Model accuracy = ",metrics.accuracy_score(y_test,pred))