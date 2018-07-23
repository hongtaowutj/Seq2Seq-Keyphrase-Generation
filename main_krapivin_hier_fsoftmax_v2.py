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
    'glove_embedding': 'nontrainable_embeddings_sent_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_sent_fsoftmax.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/hier-full',
    'result_path': 'results/krapivin/v2/hier-full',
    'decode_path': 'results/krapivin/v2/hier-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-krapivin-hier-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
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
    'vocab_size': 10004,
    'file_name' : 'decode-krapivin-hier-fsoftmax-v2',
    'weights' : 'sts-kp20k-hier-fsoftmax-v2.01-1.46.check'
   
  
}

if __name__ == '__main__':

    '''   
    
    import preprocessor_krapivin
    preprocessor_krapivin.transform_sent_fsoftmax_v2(config)
    
    
    import decoder_krapivin_hier_fsoftmax_v2
    decoder_krapivin_hier_fsoftmax_v2.decoder(config)

    '''
    

    import evaluator_krapivin
    evaluator_krapivin.evaluator(config)  

    import read_kp_krapivin
    read_kp_krapivin.reader(config)

    
    
    