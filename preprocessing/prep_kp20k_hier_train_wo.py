#!/usr/bin/anaconda3/bin/python
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from math import log
import json
from collections import OrderedDict
from prep_moduls_wo import *
import nltk
nltk.data.path.append('/home/TUE/inimah/nltk_data')
sno = nltk.stem.SnowballStemmer('english')

DATA = 'data'
eof_punc = ['.','?','!',',',';','-','_','+', '#']
punc = [x for x in string.punctuation if x not in eof_punc]

unlisted_punct = ['-', '_', '+', '#']
punct = ''.join([p for p in string.punctuation if p not in unlisted_punct])


def cleaning_text(text, punc):
    
    text = clean(striphtml(text)).lstrip().rstrip().lower()
    # remove number from text data
    #text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '', text)
    text = preprocess(text)
    # remove punctuation from 
    punct_list = str.maketrans({key: None for key in punc})
    text = text.translate(punct_list)
    text = text.replace('\n', '')
    text = text.lstrip().rstrip()
    text = text.lower()
    text = sno.stem(text)
    
    return text    


if __name__ == '__main__':

    training_data = []
    for line in open(os.path.join(DATA,'kp20k_training.json'), 'r'):
        training_data.append(json.loads(line))


    # tokenized data splitted into sentences
    tokenized_train_data = OrderedDict()
    for i, data in enumerate(training_data):
        
        clean_title = cleaning_text(data['title'], punc)
        clean_abstract = cleaning_text(data['abstract'], punc)
        clean_keywords = cleaning_text(data['keyword'], punc)

        sent_abstract = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean_abstract)
        title_tokens = tokenizeWords(clean_title)
        abstract_tokens = []
        for sent in sent_abstract:
            tokenized_abstract = tokenizeWords(sent)
            if len(tokenized_abstract) > 3:
                abstract_tokens.append(tokenized_abstract)
        keys = clean_keywords.split(';')
        keywords_tokens = []
        for kp in keys:
            keywords_tokens.append(tokenizeWords(kp))
        tokenized_train_data[i] = [title_tokens, abstract_tokens, keywords_tokens]

    savePickle(tokenized_train_data, os.path.join(DATA,'tokenized_train_data_hier'))
   

