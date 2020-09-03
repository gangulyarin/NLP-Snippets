import PyPDF2
import re
from bs4 import BeautifulSoup
import unicodedata
import string

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    #[s.extract() for s in soup['iframe', 'script']]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9!"#$%&\'()*+, -./:;<=>?@[\]^_`{|}~ ]' if not remove_digits else r'[^a-zA-z ]'
    text = re.sub(pattern, '', text)
    text = re.sub(' +',' ',text)
    return text

def clean_Text(text, remove_digits=False):
    text = strip_html_tags(text)
    text = remove_accented_chars(text)
    text = remove_special_characters(text,remove_digits)
    return text

pdfFileObj = open('input/pennymac.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)
#print(pageObj.extractText())
print(clean_Text(pageObj.extractText(),False))

# closing the pdf file object
pdfFileObj.close()