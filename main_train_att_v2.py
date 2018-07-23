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
	'result_path': 'results/kp20k/v2/sts-att',
	'decode_path': 'results/kp20k/v2/sts-att/decoding',
	'preprocessed_v2': 'results/kp20k/v2/data',
	'preprocessed_data': 'results/kp20k/v2/data',
	'glove_path': 'results/kp20k/v2/sts',
    'glove_embedding': 'nontrainable_embeddings.pkl',
    'oov_embedding': 'trainable_embeddings.pkl',

    'decoded_files': 'keyphrases-beam-sts-kp20k-att-v2',
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
	'file_name' : 'sts-kp20k-att-v2',
	'weights' : 'sts-kp20k-att-v2.12-11.67.check'
  
}

if __name__ == '__main__':

	'''

	import preprocessor

	import trainer_att_v2
	trainer_att_v2.trainer(config)

	import eval_softmax_att
	eval_softmax_att.evaluate(config)


	
	

	import decoder_att_v2
	decoder_att_v2.decoder(config)
	
	

	import evaluator
	evaluator.evaluator(config)

	
	'''

	import unigrams_evaluator
	unigrams_evaluator.evaluator(config)

	import read_kp_kp20k
	read_kp_kp20k.reader(config)

	
