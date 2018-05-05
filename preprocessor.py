import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from datetime import datetime
import time
from math import log
import json
from collections import OrderedDict

from utils.data_preprocessing_v2 import Preprocessing
from utils.indexing import Indexing
from utils.data_connector import DataConnector
from utils.sequences_processing import SequenceProcessing


'''
For KP20K data set
'''
def preprocessing_train(params):

	data_path = params['data_path']

	training_data = []
	for line in open(os.path.join(data_path,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	train_in_text = []
	train_out_keyphrases = []

	for train_data in training_data:

		text = train_data['title'] + train_data['abstract']
		train_in_text.append(text)
		train_out_keyphrases.append(train_data['keyword'].split(';'))

	train_prep = Preprocessing()
	prep_inputs = train_prep.preprocess_in(train_in_text)
	prep_outputs = train_prep.preprocess_out(train_out_keyphrases)
	train_input_tokens = train_prep.tokenize_in(prep_inputs)
	train_output_tokens = train_prep.tokenize_out(prep_outputs)
	train_tokens = train_prep.get_all_tokens(train_input_tokens, train_output_tokens)

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', train_input_tokens)
	train_in_connector.save_numpys()
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', train_output_tokens)
	train_out_connector.save_numpys()
	train_tokens_connector = DataConnector(data_path, 'train_tokens.npy', train_tokens)
	train_tokens_connector.save_numpys()
	

'''
For KP20K data set
'''

def preprocessing_valid(params):

	data_path = params['data_path']

	validation_data = []
	for line in open(os.path.join(data_path,'kp20k_validation.json'), 'r'):
		validation_data.append(json.loads(line))


	valid_in_text = []
	valid_out_keyphrases = []

	for valid_data in validation_data:

		text = valid_data['title'] + valid_data['abstract']
		valid_in_text.append(text)
		valid_out_keyphrases.append(valid_data['keyword'].split(';'))

	valid_prep = Preprocessing()
	prep_inputs = valid_prep.preprocess_in(valid_in_text)
	prep_outputs = valid_prep.preprocess_out(valid_out_keyphrases)
	valid_input_tokens = valid_prep.tokenize_in(prep_inputs)
	valid_output_tokens = valid_prep.tokenize_out(prep_outputs)
	valid_tokens = valid_prep.get_all_tokens(valid_input_tokens, valid_output_tokens)

	valid_in_connector = DataConnector(data_path, 'valid_input_tokens.npy', valid_input_tokens)
	valid_in_connector.save_numpys()
	valid_out_connector = DataConnector(data_path, 'valid_output_tokens.npy', valid_output_tokens)
	valid_out_connector.save_numpys()
	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.npy', valid_tokens)
	valid_tokens_connector.save_numpys()


'''
For KP20K data set
'''
def preprocessing_test(params):

	data_path = params['data_path']

	testing_data = []
	for line in open(os.path.join(data_path,'kp20k_testing.json'), 'r'):
		testing_data.append(json.loads(line))

	test_in_text = []
	test_out_keyphrases = []

	for test_data in testing_data:

		text = test_data['title'] + test_data['abstract']
		test_in_text.append(text)
		test_out_keyphrases.append(test_data['keyword'].split(';'))

	test_prep = Preprocessing()
	prep_inputs = test_prep.preprocess_in(test_in_text)
	prep_outputs = test_prep.preprocess_out(test_out_keyphrases)
	test_input_tokens = test_prep.tokenize_in(prep_inputs)
	test_output_tokens = test_prep.tokenize_out(prep_outputs)
	test_tokens = test_prep.get_all_tokens(test_input_tokens, test_output_tokens)

	test_in_connector = DataConnector(data_path, 'test_input_tokens.npy', test_input_tokens)
	test_in_connector.save_numpys()
	test_out_connector = DataConnector(data_path, 'test_output_tokens.npy', test_output_tokens)
	test_out_connector.save_numpys()
	test_tokens_connector = DataConnector(data_path, 'test_tokens.npy', test_tokens)
	test_tokens_connector.save_numpys()

def indexing_data(params):

	data_path = params['data_path']

	'''
	read all tokens from training, validation, and testing set
	to create vocabulary index of word tokens
	'''

	train_tokens_connector = DataConnector(data_path, 'train_tokens.npy', data=None)
	train_tokens_connector.read_numpys()
	train_tokens = train_tokens_connector.read_file

	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.npy', data=None)
	valid_tokens_connector.read_numpys()
	valid_tokens = valid_tokens_connector.read_file

	test_tokens_connector = DataConnector(data_path, 'test_tokens.npy', data=None)
	test_tokens_connector.read_numpys()
	test_tokens = test_tokens_connector.read_file

	#all_tokens = train_tokens + valid_tokens + test_tokens
	all_tokens = np.concatenate((train_tokens, valid_tokens, test_tokens))

	indexing = Indexing(all_tokens, data_path)
	indexing.vocabulary_indexing()
	indexing.save_files()
	indices_words = indexing.indices_words
	words_indices = indexing.words_indices
	term_freq = indexing.term_freq
	print("vocabulary size: %s"%len(indices_words))


def transform_train(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	x_in_connector = DataConnector(data_path, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()


def transform_valid(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)
	
	x_in_connector = DataConnector(data_path, 'X_valid.npy', X_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_out.npy', y_valid_out)
	y_out_connector.save_numpys()


def transform_test(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file


	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_test.npy', X_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_out.npy', y_test_out)
	y_out_connector.save_numpys()


def pair_train(params):

	data_path = params['data_path']

	'''
	read training set

	'''
	x_train_connector = DataConnector(data_path, 'X_train.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_in.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_out.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pairing_data(X_train, y_train_in, y_train_out)

	x_pair_train = np.array(x_pair_train)
	x_pair_train = x_pair_train.reshape((x_pair_train.shape[0], x_pair_train.shape[2]))
	y_pair_train_in = np.array(y_pair_train_in)
	y_pair_train_in = y_pair_train_in.reshape((y_pair_train_in.shape[0], y_pair_train_in.shape[2]))
	y_pair_train_out = np.array(y_pair_train_out)
	y_pair_train_out = y_pair_train_out.reshape((y_pair_train_out.shape[0], y_pair_train_out.shape[2]))

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_train.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out.npy', y_pair_train_out)
	y_out_connector.save_numpys()

def pair_valid(params):

	data_path = params['data_path']

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(data_path, 'X_valid.npy', data=None)
	x_valid_connector.read_numpys()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(data_path, 'y_valid_in.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(data_path, 'y_valid_out.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair_valid, y_pair_valid_in, y_pair_valid_out = sequences_processing.pairing_data(X_valid, y_valid_in, y_valid_out)


	x_pair_valid = np.array(x_pair_valid)
	x_pair_valid = x_pair_valid.reshape((x_pair_valid.shape[0], x_pair_valid.shape[2]))
	y_pair_valid_in = np.array(y_pair_valid_in)
	y_pair_valid_in = y_pair_valid_in.reshape((y_pair_valid_in.shape[0], y_pair_valid_in.shape[2]))
	y_pair_valid_out = np.array(y_pair_valid_out)
	y_pair_valid_out = y_pair_valid_out.reshape((y_pair_valid_out.shape[0], y_pair_valid_out.shape[2]))

	print("\nshape of x_pair_valid: %s\n"%str(x_pair_valid.shape))
	print("\nshape of y_pair_valid_in: %s\n"%str(y_pair_valid_in.shape))
	print("\nshape of y_pair_valid_out: %s\n"%str(y_pair_valid_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_valid.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_valid.npy', x_pair_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_valid_in.npy', y_pair_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_valid_out.npy', y_pair_valid_out)
	y_out_connector.save_numpys()


def pair_test(params):

	data_path = params['data_path']

	'''
	read test set

	'''
	X_test_connector = DataConnector(data_path, 'X_test.pkl', data=None)
	X_test_connector.read_pickle()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_in.pkl', data=None)
	y_test_in_connector.read_pickle()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_out.pkl', data=None)
	y_test_out_connector.read_pickle()
	y_test_out = y_test_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pairing_data(X_test, y_test_in, y_test_out)


	x_pair_test = np.array(x_pair_test)
	x_pair_test = x_pair_test.reshape((x_pair_test.shape[0], x_pair_test.shape[2]))
	y_pair_test_in = np.array(y_pair_test_in)
	y_pair_test_in = y_pair_test_in.reshape((y_pair_test_in.shape[0], y_pair_test_in.shape[2]))
	y_pair_test_out = np.array(y_pair_test_out)
	y_pair_test_out = y_pair_test_out.reshape((y_pair_test_out.shape[0], y_pair_test_out.shape[2]))

	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_in.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_out.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	