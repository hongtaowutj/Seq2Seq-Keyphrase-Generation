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
    'preprocessed_data': 'results/kp20k/v2/data',
    'glove_embedding': 'nontrainable_embeddings_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_fsoftmax.pkl',
    'model_path':'models',
    'result_path': 'results/kp20k/v2/sts-full',
    'decode_path': 'results/kp20k/v2/sts-full/decoding',

    'decoded_files': 'keyphrases-beam-sts-kp20k-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_fsoftmax.pkl',
    'y_true': 'test_output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'sts-kp20k-fsoftmax-v2',
    'weights' : 'sts-kp20k-fsoftmax-v2.02-1.31.check'
    
  
}

if __name__ == '__main__':

    '''

    print("preprocessing data...")

    import preprocessor
    # sorted vocab by TF and create vocabulary index based on most frequent --> less frequent
    preprocessor.merge_tfs_fsoftmax(config)

    # sorted based on in-vocab and out-vocab: Glove pretrained vs corpus index
    # 
    preprocessor.create_in_out_vocab_fsoftmax(config)
    preprocessor.create_embeddings_fsoftmax(config)

    print("vectorizing and padding...")

    preprocessor.transform_train_fsoftmax_v2(config)
    preprocessor.transform_valid_fsoftmax_v2(config)
    preprocessor.transform_test_fsoftmax_v2(config)

    print("pairing data...")

    preprocessor.pair_train_fsoftmax(config)
    preprocessor.pair_valid_fsoftmax(config)
    preprocessor.pair_test_fsoftmax(config)

    

    
    print("training model...")
   
    import trainer_fsoftmax_v2
    trainer_fsoftmax_v2.trainer(config)

    

    import decoder_fsoftmax_v2
    decoder_fsoftmax_v2.decoder(config)

    '''

    import evaluator
    evaluator.evaluator(config)

    import read_kp_kp20k
    read_kp_kp20k.reader(config)

    

    
