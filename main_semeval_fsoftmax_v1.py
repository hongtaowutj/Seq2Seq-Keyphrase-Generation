#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf

config = {
    'data_path': 'data/semeval2010',
    'kp20k_path': 'data/kp20k',
    'preprocessed_data': 'results/semeval2010/v1/data',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v1/sts-full',
    'result_path': 'results/semeval2010/v1/sts-full',
    'decode_path': 'results/semeval2010/v1/sts-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-semeval-sts-fsoftmax-v1',
    'idx_words': 'all_indices_words_fsoftmax.pkl',
    'words_idx': 'all_words_indices_fsoftmax.pkl',
    'x_1': 'train_input_tokens.npy',
    'x_2': 'test_input_tokens.npy',
    'y_1': 'train_output_tokens.npy',
    'y_2': 'test_output_tokens.npy',
   
    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'decode-semeval-sts-fsoftmax-v1',
    'weights' : 'sts-kp20k-fsoftmax-v1.12-1.20.check'
    

  
}

if __name__ == '__main__':
    '''

    import preprocessor_semeval

    preprocessor_semeval.transform_train_fsoftmax(config)
    preprocessor_semeval.transform_test_fsoftmax(config)
    
    
    import decoder_semeval_fsoftmax_v1
    decoder_semeval_fsoftmax_v1.decoder(config)

    
    '''

    import evaluator_semeval
    evaluator_semeval.evaluator(config)

    import read_kp_semeval
    read_kp_semeval.reader(config)

    
