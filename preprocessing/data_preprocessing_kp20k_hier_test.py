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

    tokenized_test_data = readPickle(os.path.join(DATA,'tokenized_test_data_hier'))

    indices_words = readPickle(os.path.join(DATA,'indices_words_bigru'))
    words_indices = readPickle(os.path.join(DATA,'words_indices_bigru'))

    num_words_dec = readPickle(os.path.join(DATA,'num_keyps'))
    num_words_enc = readPickle(os.path.join(DATA,'num_words_hier'))
    max_sentences = readPickle(os.path.join(DATA,'num_sents_hier'))

    '''

    test_data = OrderedDict()
    oov_test_docids = []
    for k,v in tokenized_test_data.items():
        docid = int(k)
        kps = v[2]
        sum_kps = sum(kps,[])
        len_kps = len(sum_kps)
        oov = [w for w in sum_kps if w not in indices_words.values()]
        len_oov = len(oov)
        
        if len_oov >= 0.7 * len_kps:
            oov_test_docids.append(docid)
        else:
            test_data[docid] = v
    
    savePickle(oov_test_docids, os.path.join(DATA,'oov_test_docids_hier'))
    savePickle(test_data, os.path.join(DATA,'test_data_oov_hier'))

    '''

    test_data = readPickle(os.path.join(DATA,'test_data_oov_hier'))

    encoder_length = num_words_enc # maximum sequence length (number of words) in encoder layer
    decoder_length = num_words_dec # maximum sequence length (number of words) in decoder layer

    X_test = np.zeros((len(test_data), max_sentences, encoder_length), dtype=np.int32) 
    
    test_kp_y_in = []
    test_kp_y_out = []
    all_keyphrases = []

    for i, (title,abstract,keys) in enumerate(list(test_data.values())):

        for j, sentence in enumerate(abstract):
            if j < max_sentences:
                len_sent = len(sentence) 
                if len_sent > encoder_length:
                    sent = sentence[:encoder_length]
                else:
                    sent = sentence
                for t, word in enumerate(sent):
                    if word in indices_words.values():
                        X_test[i, j, t] = words_indices[word]
                    # OOV (unknown words)
                    else: 
                        X_test[i, j, t] = words_indices['<unk>']

        kp_y_in = []
        kp_y_out = []
        keyphrases = []

        for kp in keys:

            # for true Y Key phrases
            txt_kp = " ".join(kp)
            keyphrases.append(txt_kp)

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

        test_kp_y_in.append(kp_y_in)
        test_kp_y_out.append(kp_y_out)
        all_keyphrases.append(keyphrases)


    X_test = np.array(X_test)
    y_test_in = np.array(test_kp_y_in)
    y_test_out = np.array(test_kp_y_out)


    savePickle(X_test, os.path.join(DATA,'X_test_hier_r2.pkl'))
    savePickle(y_test_in, os.path.join(DATA,'y_test_in_hier_r2.pkl'))
    savePickle(y_test_out, os.path.join(DATA,'y_test_out_hier_r2.pkl'))
    savePickle(all_keyphrases, os.path.join(DATA,'true_keyphrases_test_hier.pkl'))
