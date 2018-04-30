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
    'result_path': 'results',
    'birnn_dim': 128,
    'rnn_dim': 256,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 5,
    'batch_size': 64,
    'epoch': 100,
    'vocab_size': 1000000,
    'num_samples': 10000
  
}

if __name__ == '__main__':

    import preprocessor
    #preprocessor.preprocessing_train(config)
    preprocessor.preprocessing_valid(config)
    preprocessor.preprocessing_test(config)

    preprocessor.indexing_data(config)

    preprocessor.transform_train(config)
    preprocessor.transform_valid(config)
    preprocessor.transform_test(config)

    preprocessor.pair_train(config)
    preprocessor.pair_valid(config)

    #import trainer
    #trainer.trainer(config)
