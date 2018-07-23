#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/krapivin',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/krapivin/v2/data',
    'glove_path': 'results/kp20k/v2/sts',
    'glove_embedding': 'nontrainable_embeddings_sent.pkl',
    'oov_embedding': 'trainable_embeddings_sent.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/hier',
    'result_path': 'results/krapivin/v2/hier',
    'decode_path': 'results/krapivin/v2/hier/decoding',

    'decoded_files': 'keyphrases-beam-decode-krapivin-hier-v2',
    'idx_words': 'all_idxword_vocabulary_sent.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent.pkl',
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
    'file_name' : 'decode-krapivin-hier-v2',
    'weights' : 'sts-kp20k-hier-v2.29-14.80.check'
   
  
}

if __name__ == '__main__':

    '''
    
    
    import preprocessor_krapivin
    preprocessor_krapivin.transform_sent_v2(config)

    
    
    import decoder_krapivin_hier_v2
    decoder_krapivin_hier_v2.decoder(config)

    '''

    import evaluator_krapivin
    evaluator_krapivin.evaluator(config)  

    import read_kp_krapivin
    read_kp_krapivin.reader(config)


    
    
    