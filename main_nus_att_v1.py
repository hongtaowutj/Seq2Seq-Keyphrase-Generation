#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf

config = {
    'data_path': 'data/nus',
    'kp20k_path': 'data/kp20k',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/nus/v1/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v1/sts-att',
    'result_path': 'results/nus/v1/sts-att',
    'decode_path': 'results/nus/v1/sts-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-nus-sts-att-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
    'x_in': 'input_tokens.npy',
    'y_true': 'output_tokens.npy',


    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 159739,
    'num_samples': 10000,
    'file_name' : 'decode-nus-sts-att-v1',
    'weights' : 'sts-kp20k-att-v1.13-11.22.check'

}

if __name__ == '__main__':

    '''
    import decoder_nus_att_v1
    decoder_nus_att_v1.decoder(config)

    '''

    import evaluator_nus
    evaluator_nus.evaluator(config)

    import read_kp_nus
    read_kp_nus.reader(config)
    