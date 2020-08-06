import re
import nltk
from bs4 import BeautifulSoup
import unicodedata
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from operator import itemgetter
from nltk.collocations import BigramCollocationFinder
from nltk.collocations import BigramAssocMeasures
from gensim.summarization import keywords

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()


alice = gutenberg.sents(fileids='carroll-alice.txt')
norm_alice = [' '.join(ts) for ts in alice]

#norm_alice = list(filter(None,normalize_corpus(alice, html_stripping=False, text_lemmatization=False)))
#norm_alice = list(filter(None,normalize(alice)))

def compute_ngrams(sequence, n):
    return list(zip(*(sequence[index:] for index in range(n))))

def flatten_corpus(corpus):
    return ' '.join([document.strip() for document in corpus])

def get_top_ngrams(corpus, ngram_val=1, limit=5):
    corpus = flatten_corpus(corpus)
    tokens = nltk.word_tokenize(corpus)
    ngrams = compute_ngrams(tokens, ngram_val)
    ngrams_freq_dist = nltk.FreqDist(ngrams)
    sorted_ngrams_fd = sorted(ngrams_freq_dist.items(),
                              key=itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[0:limit]
    sorted_ngrams = [(' '.join(text), freq)
                     for text, freq in sorted_ngrams]
    return sorted_ngrams

print(get_top_ngrams(corpus=norm_alice, ngram_val=3,limit=10))

finder = BigramCollocationFinder.from_documents([item.split()
                                                for item
                                                in norm_alice])
bigram_measures = BigramAssocMeasures()

print(finder.nbest(bigram_measures.raw_freq, 10))

# Now using gensim
print("Sentence: ", norm_alice[2])
key_words = keywords(norm_alice[2], ratio=1.0, scores=True, lemmatize=True)
print([(item, round(score, 3)) for item, score in key_words][:25])