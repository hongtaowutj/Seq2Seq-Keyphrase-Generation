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
    'preprocessed_data': 'results/inspec/v2/data',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'glove_embedding': 'nontrainable_embeddings_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_fsoftmax.pkl',
    'model_path':'models',
    'result_kp20k': 'results/kp20k/v2/sts-full',
    'result_path': 'results/inspec/v2/sts-full',
    'decode_path': 'results/inspec/v2/sts-full/decoding',

    'decoded_files': 'keyphrases-beam-decode-inspec-sts-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_fsoftmax.pkl',
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
    'vocab_size': 10004,
    'file_name' : 'decode-inspec-sts-fsoftmax-v2',
    'weights' : 'sts-kp20k-fsoftmax-v2.02-1.31.check'
  
}

if __name__ == '__main__':

    '''
    import preprocessor_inspec
    
    preprocessor_inspec.transform_train_v2_fsoftmax(config)
    preprocessor_inspec.transform_valid_v2_fsoftmax(config)
    preprocessor_inspec.transform_test_v2_fsoftmax(config)
    
    
    import decoder_inspec_fsoftmax_v2
    decoder_inspec_fsoftmax_v2.decoder(config)

    '''

    import evaluator_inspec
    evaluator_inspec.evaluator(config)


    import read_kp_inspec
    read_kp_inspec.reader(config)

    
    