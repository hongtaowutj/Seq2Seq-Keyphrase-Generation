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
    'result_kp20k': 'results/kp20k/v1/sts-att-full',
    'result_path': 'results/nus/v1/sts-att-full',
    'decode_path': 'results/nus/v1/sts-att-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-nus-sts-att-fsoftmax-v1',
    'idx_words': 'all_indices_words_fsoftmax.pkl',
    'words_idx': 'all_words_indices_fsoftmax.pkl',
    'x_in': 'input_tokens.npy',
    'y_true': 'output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'decode-nus-sts-att-fsoftmax-v1',
    'weights' : 'sts-kp20k-att-fsoftmax-v1.08-11.16.check'

}

if __name__ == '__main__':

    '''
    import decoder_nus_att_fsoftmax_v1
    decoder_nus_att_fsoftmax_v1.decoder(config)

    '''

    import evaluator_nus
    evaluator_nus.evaluator(config)

    import read_kp_nus
    read_kp_nus.reader(config)
    