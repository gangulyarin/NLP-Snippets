import re
from bs4 import BeautifulSoup
import nltk
import numpy as np
import unicodedata

stopword_list = nltk.corpus.stopwords.words('english')

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup['iframe', 'script']]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def sentence_tokenize(text):
    sent_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
    sents = sent_tokenizer.tokenize(text)
    return np.array(sents)

def word_tokenize(text):
    word_tokenizer = nltk.WordPunctTokenizer()
    words = word_tokenizer.tokenize(text)
    return np.array(words)

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

#print(remove_accented_chars('Sómě Áccěntěd těxt'))
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text