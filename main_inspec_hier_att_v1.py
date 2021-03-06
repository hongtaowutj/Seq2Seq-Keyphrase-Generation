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
    'preprocessed_data': 'results/inspec/v1/data',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v1/hier-att',
    'result_path': 'results/inspec/v1/hier-att',
    'decode_path': 'results/inspec/v1/hier-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-inspec-hier-att-v1',
    'idx_words': 'all_indices_words_sent.pkl',
    'words_idx': 'all_words_indices_sent.pkl',
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
    'vocab_size': 220428,
    'num_samples': 10000,
    'file_name' : 'decode-inspec-hier-att-v1',
    'weights' : 'sts-kp20k-hier-att-v1.14-13.52.check'
  
}

if __name__ == '__main__':

    '''      
    
    import decoder_inspec_hier_att_v1
    decoder_inspec_hier_att_v1.decoder(config)

    

    '''

    import evaluator_inspec
    evaluator_inspec.evaluator(config)


    import read_kp_inspec
    read_kp_inspec.reader(config)



