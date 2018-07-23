#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
	'data_path': 'data/kp20k',
	'preprocessed_data': 'results/kp20k/v2/data',
	'preprocessed_v2': 'results/kp20k/v2/data',
    'glove_embedding': 'nontrainable_embeddings_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_fsoftmax.pkl',
	'model_path':'models',
	'result_path': 'results/kp20k/v2/sts-att-full',
	'decode_path': 'results/kp20k/v2/sts-att-full/decoding',

	'decoded_files': 'keyphrases-beam-sts-kp20k-att-fsoftmax-v2',
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
	'file_name' : 'sts-kp20k-att-fsoftmax-v2',
	'weights' : 'sts-kp20k-att-fsoftmax-v2.03-11.33.check'
  
}

if __name__ == '__main__':

	'''

	import trainer_att_fsoftmax_v2
	trainer_att_fsoftmax_v2.trainer(config)

	

	import decoder_att_fsoftmax_v2
	decoder_att_fsoftmax_v2.decoder(config)
	
	
	import evaluator
	evaluator.evaluator(config)

	

	'''

	import unigrams_evaluator
	unigrams_evaluator.evaluator(config)

	import read_kp_kp20k
	read_kp_kp20k.reader(config)
	
