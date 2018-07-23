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
    'result_kp20k': 'results/kp20k/v1/sts-att',
    'result_path': 'results/inspec/v1/sts-att',
    'decode_path': 'results/inspec/v1/sts-att/decoding',

    'decoded_files': 'keyphrases-beam-decode-inspec-att-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
    'x1': 'train_input_tokens.npy',
    'x2': 'val_input_tokens.npy',
    'x3': 'test_input_sent_tokens.npy',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 159739,
    'num_samples': 10000,
    'file_name' : 'decode-inspec-att-v1',
    'weights' : 'sts-kp20k-att-v1.13-11.22.check'
  
}

if __name__ == '__main__':

    '''        

    import decoder_inspec_att_v1
    decoder_inspec_att_v1.decoder(config)

    '''
   
    import evaluator_inspec
    evaluator_inspec.evaluator(config)


    import read_kp_inspec
    read_kp_inspec.reader(config)

   
