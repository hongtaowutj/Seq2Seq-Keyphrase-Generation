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
	'model_path':'models',
	'result_path': 'results/kp20k/v2/hier',
	'preprocessed_v2': 'results/kp20k/v2/data',
	'preprocessed_data': 'results/kp20k/v2/data',
	'decode_path': 'results/kp20k/v2/hier/decoding',

	'glove_name': 'glove.6B.100d.txt',
	'glove_w2v': 'word2vec.glove.100d.txt',
	'glove_path': 'results/kp20k/v2',
	'glove_embedding': 'nontrainable_embeddings_sent.pkl',
	'oov_embedding': 'trainable_embeddings_sent.pkl',

	'decoded_files': 'keyphrases-beam-sts-kp20k-hier-v2',
	'idx_words': 'all_idxword_vocabulary_sent.pkl',
	'words_idx': 'all_wordidx_vocabulary_sent.pkl',
	'y_true': 'test_sent_output_tokens.npy',


	'birnn_dim': 150,
	'rnn_dim': 300,
	'embedding_dim': 100,
	'encoder_length': 20,
	'decoder_length' : 8,
	'max_sents' : 20,
	'batch_size': 64,
	'epoch': 100,
	'vocab_size': 220428,
	'num_samples': 10000,
	'file_name' : 'sts-kp20k-hier-v2',
	'weights' : 'sts-kp20k-hier-v2.29-14.80.check'
   
  
}

if __name__ == '__main__':

	'''

	
	print("preprocessing data...")

	import preprocessor

	preprocessor.merge_tfs_sent(config)
	preprocessor.create_in_out_vocab_sent(config)
	preprocessor.create_embeddings_sent(config)

	print("vectorizing and padding...")

	preprocessor.transform_train_sent_v2(config)
	preprocessor.transform_valid_sent_v2(config)
	preprocessor.transform_test_sent_v2(config)

	print("pairing data...")

	preprocessor.pair_train_sent_sub(config)
	preprocessor.pair_valid_sent_sub(config)
	preprocessor.pair_test_sent_sub(config)
	
	print("training model...")

	import trainer_hier_v2
	trainer_hier_v2.trainer(config)

	

	import decoder_hier_v2
	decoder_hier_v2.decoder(config)

	'''


	import evaluator
	evaluator.evaluator(config)

   
	import read_kp_kp20k
	read_kp_kp20k.reader(config)
	

