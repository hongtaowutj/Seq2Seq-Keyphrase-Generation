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
    'preprocessed_data': 'results/nus/v2/data',
    'glove_embedding': 'nontrainable_embeddings.pkl',
    'oov_embedding': 'trainable_embeddings.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/sts-att',
    'result_path': 'results/nus/v2/sts-att',
    'decode_path': 'results/nus/v2/sts-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-nus-sts-att-v2',
    'idx_words': 'all_idxword_vocabulary.pkl',
    'words_idx': 'all_wordidx_vocabulary.pkl',
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
    'file_name' : 'decode-nus-sts-att-v2',
    'weights' : 'sts-kp20k-att-v2.12-11.67.check'

}

if __name__ == '__main__':

    '''
    import decoder_nus_att_v2
    decoder_nus_att_v2.decoder(config)

    '''

    import evaluator_nus
    evaluator_nus.evaluator(config)

    import read_kp_nus
    read_kp_nus.reader(config)
    