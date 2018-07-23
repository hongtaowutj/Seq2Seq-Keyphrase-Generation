#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/semeval2010',
    'kp20k_path': 'data/kp20k',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/semeval2010/v1/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v1/sts-att',
    'result_path': 'results/semeval2010/v1/sts-att',
    'decode_path': 'results/semeval2010/v1/sts-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-semeval-att-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
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
    'vocab_size': 159739,
    'num_samples': 10000,
    'file_name' : 'decode-semeval-att-v1',
    'weights' : 'sts-kp20k-att-v1.13-11.22.check'
  
}

if __name__ == '__main__':

    '''

    import decoder_semeval_att_v1
    decoder_semeval_att_v1.decoder(config)

    '''

    import evaluator_semeval
    evaluator_semeval.evaluator(config)

    import read_kp_semeval
    read_kp_semeval.reader(config)



      
    