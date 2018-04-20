# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import string
from string import punctuation
import re
import nltk
nltk.data.path.append('/home/TUE/inimah/nltk_data')
from nltk.corpus import stopwords
import _pickle as cPickle

eof_punc = ['.','?','!',',',';','-','_','+', '#']
punc = [x for x in string.punctuation if x not in eof_punc]

unlisted_punct = ['-', '_', '+', '#']
punct = ''.join([p for p in string.punctuation if p not in unlisted_punct])


# function to clean raw text data
def striphtml(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)

def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)

# final clean: lowercase and remove nonalpha string
def checkIsAlpha(array_of_words):

    return [w for w in array_of_words if w.isalpha()]


def preprocess(text):


    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', text)
    text = re.sub(r'/', ' / ', text) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', "", text) # eliminate numbers
    text = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', text) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
  
    return text.lower()

    
def tokenizeWords(text):
    
    #regex = re.compile('[%s]' % re.escape(punct))
    #clean_text = regex.sub('', text)
    clean_text = re.sub(r"[\-\_]+\ *", " ", text)
    tokens = clean_text.split()
    
    clean_tokens = []
    for t in tokens:
        if len(t) > 1 and len(t) < 20:
            clean_tokens.append(t.lower())
    
    #return [w for w in clean_tokens if w not in stopwords.words('english')]
    return clean_tokens

def clean_keyphrases(keyphrase_list):
  
  kp_list = []
  
  for kp in keyphrase_list:
    
    regex = re.compile('[%s]' % re.escape(punct))
    text = regex.sub('', kp)
    text = re.sub(r"[\-\_]+\ *", " ", text)
    text = text.lower()
    
    kp_list.append(text)

  return kp_list  

def indexingVocabulary(array_of_words):
    
    # frequency of word across document corpus
    tf = nltk.FreqDist(array_of_words)
    wordIndex = list(tf.keys())
    
    wordIndex.insert(0,'<pad>')
    wordIndex.append('<start>')
    wordIndex.append('<end>')
    wordIndex.append('<unk>')
    # indexing word vocabulary : pairs of (index,word)
    vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])
    
    return vocab 

# reading file in pickle format
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj

def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()
