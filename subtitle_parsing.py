import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import pandas as pd

def denoiseSubtitleSpecificProcessing(segmentList):
    parsedSegment = list()
    for segment in segmentList:
        #remove \n and concatenate segment 
        passage = str()
        entryList = list()
        for entry in segment:
            #remove any html and weird stuff
            if entry[-1] == '\n':
                #remove this escape char
                entry.replace('\n', "")
                entry = entry.strip()
                try:
                    #if you can turn it into an int then you don't want it in the segment
                    int(entry)
                except:
                    passage += " " + entry

        parsedSegment.append(passage.strip())
        
    
    return parsedSegment
 
def subtitlePreprocess(passage):

    entryList = list()
    for entry in passage:
        #denoise 
        parsedEntry = denoise_text(entry)

        #remove contradictions
        parsedEntry = replace_contractions(parsedEntry)

        #tokenize as words
        parsedEntry = nltk.word_tokenize(parsedEntry)

        #normalise 
        parsedEntry = normalize(parsedEntry)
        
        #lemmize
        parsedEntry = lemmatize_verbs(parsedEntry)

        entryList.append(parsedEntry)

    return entryList
        
#noise removal functions
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#contradictions
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

#normalize
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

#stemming and lemming 
def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas