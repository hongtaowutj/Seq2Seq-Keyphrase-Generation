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
    'glove_name': 'glove.6B.100d.txt',
    'glove_w2v': 'word2vec.glove.100d.txt',
    #'glove_path': 'data/glove.6B',
    'glove_path': 'results/kp20k/v2/sts',
    'glove_embedding': 'nontrainable_embeddings.pkl',
    'oov_embedding': 'trainable_embeddings.pkl',
    'model_path':'models',
    'result_path': 'results/kp20k/v2/sts',
    'decode_path': 'results/kp20k/v2/sts/decoding',

    'decoded_files': 'keyphrases-beam-sts-kp20k-v2',
    'idx_words': 'all_idxword_vocabulary.pkl',
    'words_idx': 'all_wordidx_vocabulary.pkl',
    'y_true': 'test_output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 300,
    'decoder_length' : 8,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 159739,
    'num_samples': 10000,
    'file_name' : 'sts-kp20k-v2',
    'weights' : 'sts-kp20k-v2.10-11.57.check'
  
  
}

if __name__ == '__main__':

    '''

    print("preprocessing data...")

    import preprocessor
    # sorted vocab by TF and create vocabulary index based on most frequent --> less frequent
    #preprocessor.merge_tfs(config)

    # sorted based on in-vocab and out-vocab: Glove pretrained vs corpus index
    # 
    #preprocessor.create_in_out_vocab(config)
    #preprocessor.create_embeddings(config)

    print("vectorizing and padding...")

    preprocessor.transform_train_v2(config)
    preprocessor.transform_valid_v2(config)
    preprocessor.transform_test_v2(config)

    print("pairing data...")

    preprocessor.pair_train_sub(config)
    preprocessor.pair_valid_sub(config)
    preprocessor.pair_test_sub(config)

    
    print("training model...")
   
    import trainer_v2
    trainer_v2.trainer(config)

    

    import decoder_v2
    decoder_v2.decoder(config)

    '''

    import evaluator
    evaluator.evaluator(config)

   
    import read_kp_kp20k
    read_kp_kp20k.reader(config)

    
    

    
