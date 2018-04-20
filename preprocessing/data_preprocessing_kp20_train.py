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

    tokenized_train_data = readPickle(os.path.join(DATA,'tokenized_train_data_wo'))
    
    indices_words = readPickle(os.path.join(DATA,'indices_words_bigru'))
    words_indices = readPickle(os.path.join(DATA,'words_indices_bigru'))

    train_data = readPickle(os.path.join(DATA,'train_data_oov'))

    num_words_dec = readPickle(os.path.join(DATA,'num_keyps'))
    num_words_enc = readPickle(os.path.join(DATA,'num_abstract'))


    encoder_length = num_words_enc # maximum sequence length (number of words) in encoder layer
    decoder_length = num_words_dec # maximum sequence length (number of words) in decoder layer

    X = np.zeros((len(train_data), encoder_length), dtype=np.int32) 
    
    train_kp_y_in = []
    train_kp_y_out = []

    for i, (title,abstract,keys) in enumerate(list(train_data.values())):

        len_doc = len(abstract)
        if len_doc > encoder_length:
          txt = abstract[:encoder_length]
        else:
          txt = abstract
        for t, word in enumerate(txt):
            if word in indices_words.values():
                X[i, t] = words_indices[word]
            # OOV (unknown words)
            else: 
                X[i, t] = words_indices['<unk>']


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

    savePickle(X_train, os.path.join(DATA,'X_train_bigru.pkl'))
    savePickle(y_train_in, os.path.join(DATA,'y_train_in_bigru.pkl'))
    savePickle(y_train_out, os.path.join(DATA,'y_train_out_bigru.pkl'))
