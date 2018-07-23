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
    'result_kp20k': 'results/kp20k/v1/hier',
    'result_path': 'results/krapivin/v1/hier',
    'decode_path': 'results/krapivin/v1/hier/decoding',

    'decoded_files': 'keyphrases-beam-decode-krapivin-hier-v1',
    'idx_words': 'all_indices_words_sent.pkl',
    'words_idx': 'all_words_indices_sent.pkl',
    'x_in': 'input_tokens.npy',
    'y_true': 'output_tokens.npy',

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
    'file_name' : 'decode-krapivin-hier-v1',
    'weights' : 'sts-kp20k-hier-v1.17-13.48.check'
   
  
}

if __name__ == '__main__':

    '''
    

    import preprocessor_krapivin
    preprocessor_krapivin.transform_sent_v1(config)
    
    import decoder_krapivin_hier_v1
    decoder_krapivin_hier_v1.decoder(config)

    '''

    import evaluator_krapivin
    evaluator_krapivin.evaluator(config)  

    import read_kp_krapivin
    read_kp_krapivin.reader(config)

    
    
    