import re
import nltk
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer
from operator import itemgetter


wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()

# Simple Normalization
def normalize(doc):
    doc = re.sub(r'[^A-Za-z\s]','',str(doc), re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    #return doc
    return filtered_tokens

# Complete Normalization
def normalize_corpus(txt, html_stripping=True, #contract_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True
                     ):

    text = str(txt)
    if(html_stripping):
        soup = BeautifulSoup(text, "html.parser")
        [s.extract() for s in soup['iframe', 'script']]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        text = stripped_text

    if(accented_char_removal):
        text = [unicodedata.normalize('NFKD', textitem).encode('ascii', 'ignore').decode('utf-8', 'ignore') for textitem in text]
    if (special_char_removal and remove_digits):
        text = re.sub(r'[^A-Za-z\s]', '', str(text), re.I | re.A)
    if (special_char_removal and not remove_digits):
        text = re.sub(r'[^A-Za-z0-9\s]', '', str(text), re.I | re.A)
    if(text_lower_case):
        text = [textitem.lower() for textitem in text ]
    if(stopword_removal):
        tokens = wpt.tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        text = ' '.join(filtered_tokens)

    if(text_lemmatization):
        tokens = wpt.tokenize(text)
        lemmatized_tokens = [wnl.lemmatize(tokens) for token in tokens]
        text = ' '.join(lemmatized_tokens)
    return wpt.tokenize(text)
