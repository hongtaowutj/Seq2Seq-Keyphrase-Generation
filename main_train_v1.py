#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
	'data_path': 'data/kp20k',
    'inspec_path': 'data/inspec',
    'krapivin_path': 'data/krapivin',
    'nus_path': 'data/nus',
    'semeval_path': 'data/semeval2010',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/kp20k/v1/data',

    'decoded_files': 'keyphrases-beam-sts-kp20k-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
    'y_true': 'test_output_tokens.npy',
    
    'model_path':'models',
    'result_path': 'results/kp20k/v1/sts',
    'decode_path': 'results/kp20k/v1/sts/decoding',
    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 159739,
    'num_samples': 10000,
    'file_name' : 'sts-kp20k-v1',
    'weights' : 'sts-kp20k-v1.18-11.37.check'
 
  
}

if __name__ == '__main__':

    '''

    print("preprocessing data...")

    import preprocessor
    
    print("vectorizing and padding...")

    preprocessor.transform_train_v1(config)
    preprocessor.transform_valid_v1(config)
    preprocessor.transform_test_v1(config)

    print("pairing data...")

    preprocessor.pair_train_sub(config)
    preprocessor.pair_valid_sub(config)
    preprocessor.pair_test_sub(config)

    
    print("training model...")
   
    import trainer_v1
    trainer_v1.trainer(config)
    
   
    import decoder_v1
    decoder_v1.decoder(config)

    

    import evaluator
    evaluator.evaluator(config)


    import read_kp_kp20k
    read_kp_kp20k.reader(config)

    '''

    import unigrams_evaluator
    unigrams_evaluator.evaluator(config)

    
    
