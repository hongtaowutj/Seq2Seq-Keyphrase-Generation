import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from keras.models import Model
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical

from utils.data_connector import DataConnector
from utils.true_keyphrases import TrueKeyphrases
from utils.decoding_fullsoftmax import DecodingSoftmax
from models.hier_seq2seq import HierarchyFullSoftmax


def decoder(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']
	decode_path = params['decode_path']
	model_path = params['model_path']
	result_path = params['result_path']
	result_kp20k = params['result_kp20k']
	file_name = params['file_name']
	weights = params['weights']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']
	max_sents = params['max_sents']
	embedding_dim = params['embedding_dim']
	birnn_dim = params['birnn_dim']
	rnn_dim = params['rnn_dim']
	vocab_size = params['vocab_size']
	batch_size = params['batch_size']
	epoch = params['epoch']
	

	'''
	Reading vocabulary dictionaries

	'''
	indices_words_connector = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	indices_words_connector.read_pickle()
	indices_words = indices_words_connector.read_file

	words_indices_connector = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	words_indices_connector.read_pickle()
	words_indices = words_indices_connector.read_file

	## merge all set into one test set for trained model

	outputs_conn = DataConnector(data_path, 'output_sent_tokens.npy', data=None)
	outputs_conn.read_numpys()
	outputs = outputs_conn.read_file


	y_test_true = outputs

	print("Ground truth of keyphrases shape: %s"%str(y_test_true.shape)) # input for encoder
	sys.stdout.flush()

	

	# non-paired data set


	X_connector = DataConnector(preprocessed_data, 'X_sent_pad_fsoftmax.npy', data=None)
	X_connector.read_numpys()
	X_in = X_connector.read_file

	
	print("\n Non-paired test set: \n")
	sys.stdout.flush()
	print("X (input for encoder) shape: %s"%str(X_in.shape)) # input for encoder
	sys.stdout.flush()
	

	'''
	Decoder model for inference stage
	Return: generated keyphrases
	'''

	full_softmax = HierarchyFullSoftmax(encoder_length=encoder_length, decoder_length=decoder_length, max_sents=max_sents, embedding_dim=embedding_dim, birnn_dim=birnn_dim, rnn_dim=rnn_dim, vocab_size=vocab_size, filepath=result_kp20k, filename=file_name, batch_train_iter=None, batch_val_iter=None, batch_size=None, steps_epoch=None, val_steps=None, epochs=None)

	# skeleton of model architecture
	full_softmax.train_hier_seq2seq()
	
	'''
	Model for retrieving softmax probability
	Return: softmax probability of prediction layer
	'''

	full_softmax.predict_seq2seq(weights)
	encoder_model = full_softmax.encoder_model
	# 1. Prediction model after being trained on sampled softmax setting
	predict_softmax_model = full_softmax.prediction_model

	'''

	Inference stage
	Model: layers from prediction model and decoder model
	Inference (text generation) approach: 
	1. One best search decoding (Greedy search): 
	   Return one best (top) probable word sequence, from joint probability of words within decoder time steps (decoder sequence length)
	2. N-Beam search decoding: 
	   Return N-top best most probable word sequences, by utilizing beam tree search per time steps and joint probability within decoder time steps (decoder sequence length)

	'''

	decoder_model=full_softmax.create_decoder_model()

	
	# transform tokenized y_true (ground truth of keyphrases) into full sentences / keyphrases
	keyphrases_transform =  TrueKeyphrases(y_test_true)
	keyphrases_transform.get_true_keyphrases()
	keyphrases_transform.get_stat_keyphrases()
	y_true = keyphrases_transform.y_true
	max_kp_num = keyphrases_transform.max_kp_num
	mean_kp_num = keyphrases_transform.mean_kp_num
	std_kp_num = keyphrases_transform.std_kp_num

	print("Maximum number of key phrases per document in corpus: %s" %max_kp_num)
	sys.stdout.flush()
	print("Average number of key phrases per document in corpus: %s" %mean_kp_num)
	sys.stdout.flush()
	print("Standard Deviation of number of key phrases per document in corpus: %s" %std_kp_num)
	sys.stdout.flush()

	# round up function for computing beam width 
	def roundup(x):
		return x if x % 5 == 0 else x + 5 - x % 5

	beam_width = int(roundup(mean_kp_num + (3 * std_kp_num)))
	num_hypotheses = beam_width
	print("\nBeam width: %s\n" %beam_width)
	sys.stdout.flush()

	inference_mode = DecodingSoftmax(encoder_model=encoder_model, decoder_model=decoder_model, indices_words=indices_words, words_indices=words_indices, enc_in_seq=None, decoder_length=decoder_length, rnn_dim=rnn_dim, beam_width=beam_width, num_hypotheses=num_hypotheses, filepath=decode_path, filename=file_name)

	t0_1 = time.time()
	print("Start beam decoding...")
	sys.stdout.flush()

	beam_keyphrases = inference_mode.beam_decoder(X_in[:500])
	
	beam_decode_connector = DataConnector(decode_path, 'beam_kp-hier-%s.npy'%(file_name), beam_keyphrases)
	beam_decode_connector.save_numpys()

	t1_1 = time.time()
	print("Beam decoding is done in %.3fsec" % (t1_1 - t0_1))
	sys.stdout.flush()


