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
from utils.data_iterator import Dataiterator
from utils.true_keyphrases import TrueKeyphrases
from utils.decoding import Decoding
from models.hier_seq2seq_sampled_softmax import HierarchySampledSoftmax


def trainer(params):

	try:

		data_path = params['data_path']
		preprocessed_data = params['preprocessed_data']
		preprocessed_v2 = params['preprocessed_v2']
		model_path = params['model_path']
		result_path = params['result_path']
		file_name = params['file_name']

		encoder_length = params['encoder_length']
		decoder_length = params['decoder_length']
		max_sents = params['max_sents']
		embedding_dim = params['embedding_dim']
		birnn_dim = params['birnn_dim']
		rnn_dim = params['rnn_dim']
		vocab_size = params['vocab_size']
		num_samples= params['num_samples']
		batch_size = params['batch_size']
		epoch = params['epoch']

		'''
		Reading vocabulary dictionaries

		'''
		indices_words_connector = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
		indices_words_connector.read_pickle()
		indices_words = indices_words_connector.read_file

		words_indices_connector = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
		words_indices_connector.read_pickle()
		words_indices = words_indices_connector.read_file



		'''
		Reading X, y pair data set for training and validating model

		'''
		# 1. training set

		X_train_connector = DataConnector(preprocessed_data, 'x_pair_train_sent.npy', data=None)
		X_train_connector.read_numpys()
		X_train = X_train_connector.read_file

		y_train_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in_sent.npy', data=None)
		y_train_in_connector.read_numpys()
		y_train_in = y_train_in_connector.read_file

		y_train_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out_sent.npy', data=None)
		y_train_out_connector.read_numpys()
		y_train_out = y_train_out_connector.read_file

		print("\n X,y pair of training set: \n")
		sys.stdout.flush()
		print("X (input for encoder) shape: %s"%str(X_train.shape)) # input for encoder
		sys.stdout.flush()
		print("y_in (input for decoder) shape: %s"%str(y_train_in.shape)) # input for decoder
		sys.stdout.flush()
		print("y_out (output for decoder) shape: %s\n\n"%str(y_train_out.shape)) # output for decoder
		sys.stdout.flush()

		# 2. validation set

		# pair data set

		X_valid_pair_connector = DataConnector(preprocessed_data, 'x_pair_valid_sent.npy', data=None)
		X_valid_pair_connector.read_numpys()
		X_valid_pair = X_valid_pair_connector.read_file

		y_valid_in_pair_connector = DataConnector(preprocessed_data, 'y_pair_valid_in_sent.npy', data=None)
		y_valid_in_pair_connector.read_numpys()
		y_valid_in_pair = y_valid_in_pair_connector.read_file

		y_valid_out_pair_connector = DataConnector(preprocessed_data, 'y_pair_valid_out_sent.npy', data=None)
		y_valid_out_pair_connector.read_numpys()
		y_valid_out_pair = y_valid_out_pair_connector.read_file

		print("\n X, y pair of validation set: \n")
		sys.stdout.flush()
		print("X (input for encoder) shape: %s"%str(X_valid_pair.shape)) # input for encoder
		sys.stdout.flush()
		print("y_in (input for decoder) shape: %s"%str(y_valid_in_pair.shape)) # input for decoder
		sys.stdout.flush()
		print("y_out (output for decoder) shape: %s\n\n"%str(y_valid_out_pair.shape)) # output for decoder
		sys.stdout.flush()



		steps_epoch = len(X_train)/batch_size
		batch_train_iter = Dataiterator(X_train, y_train_in, y_train_out, decoder_dim=rnn_dim, batch_size=batch_size)

		val_steps = len(X_valid_pair)/batch_size
		batch_val_iter = Dataiterator(X_valid_pair, y_valid_in_pair, y_valid_out_pair, decoder_dim=rnn_dim, batch_size=batch_size)


	except:
		raise

	'''
	1. Initiate model for training Seq2Seq with sampled softmax layer
	2. Compile with sampled softmax training loss, as an underestimate of full softmax loss
	3. Train with per-batch samples

	'''

	

	sampled_softmax = HierarchySampledSoftmax(encoder_length, decoder_length, max_sents, embedding_dim, birnn_dim, rnn_dim, vocab_size, num_samples, result_path, file_name, batch_train_iter, batch_val_iter, batch_size, steps_epoch, val_steps, epoch)

	'''
	Train model with sampled softmax layer 
	Return: LOSS in training stage (an underestimate of full softmax)
	'''

	print(str(datetime.now()))
	sys.stdout.flush()

	sampled_softmax.train_hier_sampled_softmax()
	sampled_softmax.compile_()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Training hierarchical model with approximate softmax...")
	sys.stdout.flush()

	sampled_softmax.train_()

	t1 = time.time()
	print("training is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

	sampled_softmax.plot_()

