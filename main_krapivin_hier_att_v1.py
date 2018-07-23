#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/krapivin',
    'kp20k_path': 'data/kp20k',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/krapivin/v1/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v1/hier-att',
    'result_path': 'results/krapivin/v1/hier-att',
    'decode_path': 'results/krapivin/v1/hier-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-krapivin-hier-att-v1',
    'idx_words': 'all_indices_words_sent.pkl',
    'words_idx': 'all_words_indices_sent.pkl',
    'x_in': 'input_tokens.npy',
    'y_true': 'output_sent_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 20,
    'decoder_length' : 8,
    'max_sents' : 20,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 220428,
    'num_samples': 10000,
    'file_name' : 'decode-krapivin-hier-att-v1',
    'weights' : 'sts-kp20k-hier-att-v1.14-13.52.check'
  
}

if __name__ == '__main__':

    '''

    
    import decoder_krapivin_hier_att_v1
    decoder_krapivin_hier_att_v1.decoder(config)

    '''

    import evaluator_krapivin
    evaluator_krapivin.evaluator(config)  

    import read_kp_krapivin
    read_kp_krapivin.reader(config)
    
    
    