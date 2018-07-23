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
    'preprocessed_data': 'results/krapivin/v2/data',
    'glove_embedding': 'nontrainable_embeddings_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_fsoftmax.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/sts-full',
    'result_path': 'results/krapivin/v2/sts-full',
    'decode_path': 'results/krapivin/v2/sts-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-krapivin-sts-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_fsoftmax.pkl',
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
    'file_name' : 'decode-krapivin-sts-fsoftmax-v2',
    'weights' : 'sts-kp20k-fsoftmax-v2.02-1.31.check'
    
  
}

if __name__ == '__main__':

    '''
    import preprocessor_krapivin
    preprocessor_krapivin.transform_v2_fsoftmax(config) 

    
        
    import decoder_krapivin_fsoftmax_v2
    decoder_krapivin_fsoftmax_v2.decoder(config)

    '''

    import evaluator_krapivin
    evaluator_krapivin.evaluator(config)   

    import read_kp_krapivin
    read_kp_krapivin.reader(config)
    
    