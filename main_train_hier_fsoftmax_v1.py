#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/kp20k',
    'model_path':'models',
    'result_path': 'results/kp20k/v1/hier-full',
    'decode_path': 'results/kp20k/v1/hier-full/decoding',
    'preprocessed_data': 'results/kp20k/v1/data',
    'preprocessed_v2': 'results/kp20k/v2/data',

    'decoded_files': 'keyphrases-beam-sts-kp20k-hier-fsoftmax-v1',
    'idx_words': 'all_indices_words_sent_fsoftmax.pkl',
    'words_idx': 'all_words_indices_sent_fsoftmax.pkl',
    'y_true': 'test_sent_output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 20,
    'decoder_length' : 8,
    'max_sents' : 20,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'sts-kp20k-hier-fsoftmax-v1',
    'weights' : 'sts-kp20k-hier-fsoftmax-v1.11-1.38.check'
   
  
}

if __name__ == '__main__':

    '''

    print("preprocessing data...")

    import preprocessor


    print("vectorizing and padding...")

    preprocessor.transform_train_sent_fsoftmax_v1(config)
    preprocessor.transform_valid_sent_fsoftmax_v1(config)
    preprocessor.transform_test_sent_fsoftmax_v1(config)

    print("pairing data...")

    preprocessor.pair_train_sent_fsoftmax(config)
    preprocessor.pair_valid_sent_fsoftmax(config)
    preprocessor.pair_test_sent_fsoftmax(config)

    
    print("training model...")

    import trainer_hier_fsoftmax_v1
    trainer_hier_fsoftmax_v1.trainer(config)

    

    import decoder_hier_fsoftmax_v1
    decoder_hier_fsoftmax_v1.decoder(config)

    '''

    import evaluator
    evaluator.evaluator(config)


    import read_kp_kp20k
    read_kp_kp20k.reader(config)

