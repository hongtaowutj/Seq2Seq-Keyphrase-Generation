#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
	'data_path': 'data/kp20k',
	'preprocessed_v2': 'results/kp20k/v2/data',
	'preprocessed_data': 'results/kp20k/v1/data',
	'model_path':'models',
	'result_path': 'results/kp20k/v1/sts-att',
	'decode_path': 'results/kp20k/v1/sts-att/decoding',

	'decoded_files': 'keyphrases-beam-sts-kp20k-att-v1',
	'idx_words': 'all_indices_words.pkl',
	'words_idx': 'all_words_indices.pkl',
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
	'file_name' : 'sts-kp20k-att-v1',
	'weights' : 'sts-kp20k-att-v1.13-11.22.check'
  
}

if __name__ == '__main__':

	'''

	import trainer_att_v1
	trainer_att_v1.trainer(config)


	import decoder_att_v1
	decoder_att_v1.decoder(config)

	import eval_softmax_att
	eval_softmax_att.evaluate(config)


	'''	
	

	import evaluator
	evaluator.evaluator(config)

	import read_kp_kp20k
	read_kp_kp20k.reader(config)

	