#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
	'data_path': 'data/kp20k',
	'preprocessed_data': 'results/kp20k/v1/data',
	'preprocessed_v2': 'results/kp20k/v2/data',

	'decoded_files': 'keyphrases-beam-sts-kp20k-att-fsoftmax-v1',
	'idx_words': 'all_indices_words_fsoftmax.pkl',
	'words_idx': 'all_words_indices_fsoftmax.pkl',
	'y_true': 'test_output_tokens.npy',
   
	'model_path':'models',
	'result_path': 'results/kp20k/v1/sts-att-full',
	'decode_path': 'results/kp20k/v1/sts-att-full/decoding',
	'birnn_dim': 150,
	'rnn_dim': 300,
	'embedding_dim': 100,
	'encoder_length': 300,
	'decoder_length' : 8,
	'batch_size': 128,
	'epoch': 100,
	'vocab_size': 10004,
	'file_name' : 'sts-kp20k-att-fsoftmax-v1',
	'weights' : 'sts-kp20k-att-fsoftmax-v1.08-11.16.check'
}

if __name__ == '__main__':

	'''

	import trainer_att_fsoftmax_v1
	trainer_att_fsoftmax_v1.trainer(config)

	

	import decoder_att_fsoftmax_v1
	decoder_att_fsoftmax_v1.decoder(config)
	
	

	'''

	import evaluator
	evaluator.evaluator(config)


	import read_kp_kp20k
	read_kp_kp20k.reader(config)

	
	
