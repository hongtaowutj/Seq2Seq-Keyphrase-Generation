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
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/semeval2010/v2/data',
    'glove_embedding': 'nontrainable_embeddings_sent_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_sent_fsoftmax.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/hier-full',
    'result_path': 'results/semeval2010/v2/hier-full',
    'decode_path': 'results/semeval2010/v2/hier-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-semeval-hier-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
    'x_1': 'train_input_tokens.npy',
    'x_2': 'test_input_tokens.npy',
    'y_1': 'train_sent_output_tokens.npy',
    'y_2': 'test_sent_output_tokens.npy',


    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 20,
    'decoder_length' : 8,
    'max_sents' : 20,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'decode-semeval-hier-fsoftmax-v2',
    'weights' : 'sts-kp20k-hier-fsoftmax-v2.01-1.46.check'

}

if __name__ == '__main__':

    '''

    import preprocessor_semeval

    preprocessor_semeval.transform_train_sent_fsoftmax_v2(config)
    preprocessor_semeval.transform_test_sent_fsoftmax_v2(config)


    import decoder_semeval_hier_fsoftmax_v2
    decoder_semeval_hier_fsoftmax_v2.decoder(config)

    
    '''

    import evaluator_semeval
    evaluator_semeval.evaluator(config)

    import read_kp_semeval
    read_kp_semeval.reader(config)
    
    
