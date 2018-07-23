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
from models.hier_seq2seq_sampled_softmax_v2 import HierarchySampledSoftmax


def trainer(params):

	try:

		data_path = params['data_path']
		preprocessed_data = params['preprocessed_data']
		glove_path = params['glove_path']
		#glove_name = params['glove_name']
		glove_embed = params['glove_embedding']
		oov_embed = params['oov_embedding']
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
		indices_words_connector = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', data=None)
		indices_words_connector.read_pickle()
		indices_words = indices_words_connector.read_file

		words_indices_connector = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent.pkl', data=None)
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


		'''
		Data iterator: preparing per batch training set
		
		INPUTS: 

		x_train (sequence in encoder). dimension shape ( #examples, encoder_length )

		y_train_in (y_true labels to be exposed to the decoder layer as a part of "teacher forcing" method). dimension shape ( #examples, decoder_length )

		y_train_out (y_true labels as the output projection of decoder layer -- need to be provided to calculate entropy of prediction vs. ground truth)

		OUTPUTS: 
		Per batch training examples through data iterator and generator

		x_train, y_train_in, states, labels, y_output

		states : numpy zeros array with dimension (#examples, #decoder dimension) --> will be used as the initial state of decoder

		labels: 3D shape of y_train_out. Needed as input for sampled softmax layer. dimension shape ( #examples, decoder_length, 1 )

		y_output: list format of labels --> since we use one-step decoder and sampled softmax projection. ( decoder_length, #examples, 1 )


		'''

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

	glove_embedding_conn = DataConnector(preprocessed_data, glove_embed, data=None)
	glove_embedding_conn.read_pickle()
	pretrained_embedding = glove_embedding_conn.read_file

	print("pretrained_embedding shape: %s"%str(pretrained_embedding.shape))
	print("pretrained_embedding [0][:10]: %s"%str(pretrained_embedding[0,:10]))
	print("pretrained_embedding [1][:10]: %s"%str(pretrained_embedding[1,:10]))

	oov_embedding_conn = DataConnector(preprocessed_data, oov_embed, data=None)
	oov_embedding_conn.read_pickle()
	oov_embedding = oov_embedding_conn.read_file

	print("oov_embedding shape: %s"%str(oov_embedding.shape))
	print("oov_embedding [0][:10]: %s"%str(oov_embedding[0,:10]))
	print("oov_embedding [1][:10]: %s"%str(oov_embedding[1,:10]))
	print("oov_embedding [2][:10]: %s"%str(oov_embedding[2,:10]))

	sampled_softmax = HierarchySampledSoftmax(encoder_length, decoder_length, max_sents, embedding_dim, birnn_dim, rnn_dim, vocab_size, num_samples, result_path, file_name, batch_train_iter, batch_val_iter, batch_size, steps_epoch, val_steps, epoch)

	'''
	Train model with sampled softmax layer 
	Return: LOSS in training stage (an underestimate of full softmax)
	'''

	print(str(datetime.now()))
	sys.stdout.flush()

	sampled_softmax.train_hier_sampled_softmax(pretrained_embedding, oov_embedding)
	sampled_softmax.compile_()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Training model with approximate softmax...")
	sys.stdout.flush()

	sampled_softmax.train_()

	t1 = time.time()
	print("training is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

	sampled_softmax.plot_()

