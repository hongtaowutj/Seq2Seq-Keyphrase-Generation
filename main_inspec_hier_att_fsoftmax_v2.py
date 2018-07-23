#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/inspec',
    'kp20k_path': 'data/kp20k',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/hier-att-full',
    'result_path': 'results/inspec/v2/hier-att-full',
    'decode_path': 'results/inspec/v2/hier-att-full/decoding',
    'glove_embedding': 'nontrainable_embeddings_sent_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_sent_fsoftmax.pkl',

    'decoded_files': 'keyphrases-beam-decode-semeval-hier-att-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
    'x1': 'train_input_tokens.npy',
    'x2': 'val_input_tokens.npy',
    'x3': 'test_input_sent_tokens.npy',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy',


    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 20,
    'decoder_length' : 8,
    'max_sents' : 20,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'decode-inspec-hier-att-fsoftmax-v2',
    'weights' : 'sts-kp20k-hier-att-fsoftmax-v2.03-12.23.check'
  
}

if __name__ == '__main__':

    '''
       
    import decoder_inspec_hier_att_fsoftmax_v2
    decoder_inspec_hier_att_fsoftmax_v2.decoder(config)

    '''

    import evaluator_inspec
    evaluator_inspec.evaluator(config)


    import read_kp_inspec
    read_kp_inspec.reader(config)
    
