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

    tokenized_valid_data = readPickle(os.path.join(DATA,'tokenized_valid_data_hier'))

    indices_words = readPickle(os.path.join(DATA,'indices_words_bigru'))
    words_indices = readPickle(os.path.join(DATA,'words_indices_bigru'))

    num_words_dec = readPickle(os.path.join(DATA,'num_keyps'))
    num_words_enc = readPickle(os.path.join(DATA,'num_words_hier'))
    max_sentences = readPickle(os.path.join(DATA,'num_sents_hier'))
    '''
   
    valid_data = OrderedDict()
    oov_valid_docids = []
    for k,v in tokenized_valid_data.items():
        docid = int(k)
        kps = v[2]
        sum_kps = sum(kps,[])
        len_kps = len(sum_kps)
        oov = [w for w in sum_kps if w not in indices_words.values()]
        len_oov = len(oov)
        
        if len_oov >= 0.7 * len_kps:
            oov_valid_docids.append(docid)
        else:
            valid_data[docid] = v


    savePickle(oov_valid_docids, os.path.join(DATA,'oov_valid_docids_hier'))
    savePickle(valid_data, os.path.join(DATA,'valid_data_oov_hier'))
    '''

    valid_data = readPickle(os.path.join(DATA,'valid_data_oov_hier'))

    encoder_length = num_words_enc # maximum sequence length (number of words) in encoder layer
    decoder_length = num_words_dec # maximum sequence length (number of words) in decoder layer

    X_valid = np.zeros((len(valid_data), max_sentences, encoder_length), dtype=np.int32) 
    
    valid_kp_y_in = []
    valid_kp_y_out = []

    for i, (title,abstract,keys) in enumerate(list(valid_data.values())):

        for j, sentence in enumerate(abstract):
            if j < max_sentences:
                len_sent = len(sentence) 
                if len_sent > encoder_length:
                    sent = sentence[:encoder_length]
                else:
                    sent = sentence
                for t, word in enumerate(sent):
                    if word in indices_words.values():
                        X_valid[i, j, t] = words_indices[word]
                    # OOV (unknown words)
                    else: 
                        X_valid[i, j, t] = words_indices['<unk>']

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

        valid_kp_y_in.append(kp_y_in)
        valid_kp_y_out.append(kp_y_out)

    

    X_valid = np.array(X_valid)
    y_valid_in = np.array(valid_kp_y_in)
    y_valid_out = np.array(valid_kp_y_out)

   
    savePickle(X_valid, os.path.join(DATA,'X_valid_hier.pkl'))
    savePickle(y_valid_in, os.path.join(DATA,'y_valid_in_hier.pkl'))
    savePickle(y_valid_out, os.path.join(DATA,'y_valid_out_hier.pkl'))
