#!/usr/bin/anaconda3/bin/python
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from math import log
from collections import OrderedDict
from prep_moduls_wo import *

DATA = 'data'

def roundup(x):
    return x if x % 5 == 0 else x + 5 - x % 5

if __name__ == '__main__':
    
    # without stopword removal

    tokenized_train_data = readPickle(os.path.join(DATA,'tokenized_train_data_hier'))

    indices_words = readPickle(os.path.join(DATA,'indices_words_bigru'))
    words_indices = readPickle(os.path.join(DATA,'words_indices_bigru'))

    num_words_dec = readPickle(os.path.join(DATA,'num_keyps'))
    num_words_enc = readPickle(os.path.join(DATA,'num_words_hier'))
    max_sentences = readPickle(os.path.join(DATA,'num_sents_hier'))

    '''
    
    train_data = OrderedDict()
    oov_train_docids = []
    for k,v in tokenized_train_data.items():
        docid = int(k)
        kps = v[2]
        sum_kps = sum(kps,[])
        len_kps = len(sum_kps)
        oov = [w for w in sum_kps if w not in indices_words.values()]
        len_oov = len(oov)
        
        if len_oov >= 0.7 * len_kps:
            oov_train_docids.append(docid)
        else:
            train_data[docid] = v

    savePickle(oov_train_docids, os.path.join(DATA,'oov_train_docids_hier'))
    savePickle(train_data, os.path.join(DATA,'train_data_oov_hier'))


    n_sents = []
    n_tokens = []
    for k, v in train_data.items():
        doc_abstract = v[1]
        n_sents.append(len(doc_abstract))
        for j, sent in enumerate(doc_abstract):
            n_tokens.append(len(sent))

    mean_words = np.mean(np.array(n_tokens))
    std_words = np.std(np.array(n_tokens))

    mean_sent = np.mean(np.array(n_sents))
    std_sent = np.std(np.array(n_sents))

    num_words = int(roundup(mean_words))
    savePickle(num_words, os.path.join(DATA,'num_words_hier'))

    num_sents = int(roundup(mean_sent))
    savePickle(num_sents, os.path.join(DATA,'num_sents_hier'))

    '''

    train_data = readPickle(os.path.join(DATA,'train_data_oov_hier'))

   
    encoder_length = num_words_enc # maximum sequence length (number of words) in encoder layer
    decoder_length = num_words_dec # maximum sequence length (number of words) in decoder layer


    X = np.zeros((len(train_data), max_sentences, encoder_length), dtype=np.int32) 
    
    train_kp_y_in = []
    train_kp_y_out = []

    for i, (title,abstract,keys) in enumerate(list(train_data.values())):

        for j, sentence in enumerate(abstract):
            if j < max_sentences:
                len_sent = len(sentence) 
                if len_sent > encoder_length:
                    sent = sentence[:encoder_length]
                else:
                    sent = sentence
                for t, word in enumerate(sent):
                    if word in indices_words.values():
                        X[i, j, t] = words_indices[word]
                    # OOV (unknown words)
                    else: 
                        X[i, j, t] = words_indices['<unk>']

        kp_y_in = []
        kp_y_out = []

        for kp in keys:

            y_in = np.zeros((1, decoder_length+1), dtype=np.int32) 
            y_out = np.zeros((1, decoder_length+1), dtype=np.int32) 

            len_kp = len(kp)

            if len_kp > decoder_length:
                txt_kp = kp[:decoder_length]
            else:
                txt_kp = kp

            txt_in = list(txt_kp)
            txt_out = list(txt_kp)
            txt_in.insert(0,'<start>')
            txt_out.append('<end>')

            for k, word in enumerate(txt_in):
                if word in indices_words.values():
                    y_in[0, k] = words_indices[word]
                else:
                    y_in[0, k] = words_indices['<unk>']

            kp_y_in.append(y_in)

            for k, word in enumerate(txt_out):
                if word in indices_words.values():
                    y_out[0, k] = words_indices[word]
                else:
                    y_out[0, k] = words_indices['<unk>']
            kp_y_out.append(y_out)

        train_kp_y_in.append(kp_y_in)
        train_kp_y_out.append(kp_y_out)
    
    X_train = np.array(X)
    y_train_in = np.array(train_kp_y_in)
    y_train_out = np.array(train_kp_y_out)

    savePickle(X_train, os.path.join(DATA,'X_train_hier.pkl'))
    savePickle(y_train_in, os.path.join(DATA,'y_train_in_hier.pkl'))
    savePickle(y_train_out, os.path.join(DATA,'y_train_out_hier.pkl'))
