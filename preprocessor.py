import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from math import log
import json
from collections import OrderedDict

from utils.data_preprocessing import Preprocessing
from utils.indexing import Indexing
from utils.data_connector import DataConnector




def preprocessing_train(params):

	data_path = params['data_path']

	training_data = []
	for line in open(os.path.join(data_path,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	train_input_tokens = []
	train_output_tokens = []
	train_tokens = []
	for train_data in training_data:

		train_in_text = train_data['title'] + train_data['abstract']
		train_out_keyphrases = train_data['keyword']

		train_prep = Preprocessing(train_in_text, train_out_keyphrases)
		train_prep.preprocess_articles()
		train_prep.tokenize_words()

		inputs_tokens = train_prep.inputs_tokens
		output_tokens = train_prep.output_tokens
		train_input_tokens.append(inputs_tokens)
		train_output_tokens.append(output_tokens)

		train_tokens.extend(inputs_tokens)
		for kp in output_tokens:
			for t in kp:
				train_tokens.append(t)

	train_in_connector = DataConnector(data_path, 'train_input_tokens.pkl', train_input_tokens)
	train_in_connector.save_pickle()
	train_out_connector = DataConnector(data_path, 'train_output_tokens.pkl', train_output_tokens)
	train_out_connector.save_pickle()
	train_tokens_connector = DataConnector(data_path, 'train_tokens.pkl', train_tokens)
	train_tokens_connector.save_pickle()



def preprocessing_valid(params):

	data_path = params['data_path']

	validation_data = []
	for line in open(os.path.join(data_path,'kp20k_validation.json'), 'r'):
		validation_data.append(json.loads(line))

	valid_input_tokens = []
	valid_output_tokens = []
	valid_tokens = []
	for valid_data in validation_data:

		valid_in_text = valid_data['title'] + valid_data['abstract']
		valid_out_keyphrases = valid_data['keyword']

		valid_prep = Preprocessing(valid_in_text, valid_out_keyphrases)
		valid_prep.preprocess_articles()
		valid_prep.tokenize_words()

		inputs_tokens = valid_prep.inputs_tokens
		output_tokens = valid_prep.output_tokens
		valid_input_tokens.append(inputs_tokens)
		valid_output_tokens.append(output_tokens)

		valid_tokens.extend(inputs_tokens)
		for kp in output_tokens:
			for t in kp:
				valid_tokens.append(t)

	valid_in_connector = DataConnector(data_path, 'valid_input_tokens.pkl', valid_input_tokens)
	valid_in_connector.save_pickle()
	valid_out_connector = DataConnector(data_path, 'valid_output_tokens.pkl', valid_output_tokens)
	valid_out_connector.save_pickle()
	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.pkl', valid_tokens)
	valid_tokens_connector.save_pickle()

def preprocessing_test(params):

	data_path = params['data_path']

	testing_data = []
	for line in open(os.path.join(data_path,'kp20k_testing.json'), 'r'):
		testing_data.append(json.loads(line))

	test_input_tokens = []
	test_output_tokens = []
	test_tokens = []
	for test_data in testing_data:

		test_in_text = test_data['title'] + test_data['abstract']
		test_out_keyphrases = test_data['keyword']

		test_prep = Preprocessing(test_in_text, test_out_keyphrases)
		test_prep.preprocess_articles()
		test_prep.tokenize_words()

		inputs_tokens = test_prep.inputs_tokens
		output_tokens = test_prep.output_tokens
		test_input_tokens.append(inputs_tokens)
		test_output_tokens.append(output_tokens)

		test_tokens.extend(inputs_tokens)
		for kp in output_tokens:
			for t in kp:
				test_tokens.append(t)

	test_in_connector = DataConnector(data_path, 'test_input_tokens.pkl', test_input_tokens)
	test_in_connector.save_pickle()
	test_out_connector = DataConnector(data_path, 'test_output_tokens.pkl', test_output_tokens)
	test_out_connector.save_pickle()
	test_tokens_connector = DataConnector(data_path, 'test_tokens.pkl', test_tokens)
	test_tokens_connector.save_pickle()

def indexing_data(params):

	data_path = params['data_path']

	'''
	read all tokens from training, validation, and testing set
	to create vocabulary index of word tokens
	'''

	train_tokens_connector = DataConnector(data_path, 'train_tokens.pkl', data=None)
	train_tokens_connector.read_pickle()
	train_tokens = train_tokens_connector.read_file

	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.pkl', data=None)
	valid_tokens_connector.read_pickle()
	valid_tokens = valid_tokens_connector.read_file

	test_tokens_connector = DataConnector(data_path, 'test_tokens.pkl', data=None)
	test_tokens_connector.read_pickle()
	test_tokens = test_tokens_connector.read_file

	all_tokens = train_tokens + valid_tokens + test_tokens

	'''

	# choose 100K most frequent words
	term_freq = nltk.FreqDist(all_tokens)
	common_words = term_freq.most_common(100000)
	arr_common = np.array(common_words)
	words = arr_common[:,0]
	'''

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

	train_in_connector = DataConnector(data_path, 'train_input_tokens.pkl', data=None)
	train_in_connector.read_pickle()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_tokens.pkl', data=None)
	train_out_connector.read_pickle()
	train_out_tokens = train_out_connector.read_file

	# create X,y pair for model training and evaluation
	X_train = np.zeros((len(train_in_tokens), encoder_length), dtype=np.int32)
	train_kp_y_in = []
	train_kp_y_out = []

	for i, (in_text, out_text) in enumerate(zip(train_in_tokens, train_out_tokens)):

		len_in_text = len(in_text)
		if len_in_text > encoder_length:
			txt = in_text[:encoder_length]
		else:
			txt = in_text

		for t, word in enumerate(txt):
			if word in indices_words.values():
				X_train[i, t] = words_indices[word]
			# OOV (unknown words)
			else: 
				X_train[i, t] = words_indices['<unk>']

		kp_y_in = []
		kp_y_out = []

		for kp in out_text:

			y_in = np.zeros((1, decoder_length+1), dtype=np.int32) 
			y_out = np.zeros((1, decoder_length+1), dtype=np.int32)

			len_kp = len(kp)

			if len_kp > decoder_length:
				txt_kp = kp[:decoder_length]
			else:
				txt_kp = kp

			txt_in = list(txt_kp)
			txt_out = list(txt_kp)
			txt_in.insert(0,'<start>')
			txt_out.append('<end>')

			for k, word in enumerate(txt_in):
				if word in indices_words.values():
					y_in[0, k] = words_indices[word]
				else:
					y_in[0, k] = words_indices['<unk>']

			kp_y_in.append(y_in)

			for k, word in enumerate(txt_out):
				if word in indices_words.values():
					y_out[0, k] = words_indices[word]
				else:
					y_out[0, k] = words_indices['<unk>']

			kp_y_out.append(y_out)

		train_kp_y_in.append(kp_y_in)
		train_kp_y_out.append(kp_y_out)

	X_train = np.array(X_train)
	y_train_in = np.array(train_kp_y_in)
	y_train_out = np.array(train_kp_y_out)

	x_in_connector = DataConnector(data_path, 'X_train.pkl', X_train)
	x_in_connector.save_pickle()
	y_in_connector = DataConnector(data_path, 'y_train_in.pkl', y_train_in)
	y_in_connector.save_pickle()
	y_out_connector = DataConnector(data_path, 'y_train_out.pkl', y_train_out)
	y_out_connector.save_pickle()


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

	valid_in_tokens_connector = DataConnector(data_path, 'valid_input_tokens.pkl', data=None)
	valid_in_tokens_connector.read_pickle()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_output_tokens.pkl', data=None)
	valid_out_tokens_connector.read_pickle()
	valid_out_tokens = valid_out_tokens_connector.read_file

	# create X,y pair for model training and evaluation
	X_valid = np.zeros((len(valid_in_tokens), encoder_length), dtype=np.int32)
	valid_kp_y_in = []
	valid_kp_y_out = []

	for i, (in_text, out_text) in enumerate(zip(valid_in_tokens, valid_out_tokens)):

		len_in_text = len(in_text)
		if len_in_text > encoder_length:
			txt = in_text[:encoder_length]
		else:
			txt = in_text

		for t, word in enumerate(txt):
			if word in indices_words.values():
				X_valid[i, t] = words_indices[word]
			# OOV (unknown words)
			else: 
				X_valid[i, t] = words_indices['<unk>']

		kp_y_in = []
		kp_y_out = []

		for kp in keys:

			y_in = np.zeros((1, decoder_length+1), dtype=np.int32) 
			y_out = np.zeros((1, decoder_length+1), dtype=np.int32) 

			len_kp = len(kp)

			if len_kp > decoder_length:
				txt_kp = kp[:decoder_length]
			else:
				txt_kp = kp

			txt_in = list(txt_kp)
			txt_out = list(txt_kp)
			txt_in.insert(0,'<start>')
			txt_out.append('<end>')

			for k, word in enumerate(txt_in):
				if word in indices_words.values():
					y_in[0, k] = words_indices[word]
				else:
					y_in[0, k] = words_indices['<unk>']

			kp_y_in.append(y_in)

			for k, word in enumerate(txt_out):
				if word in indices_words.values():
					y_out[0, k] = words_indices[word]
				else:
					y_out[0, k] = words_indices['<unk>']

			kp_y_out.append(y_out)

		valid_kp_y_in.append(kp_y_in)
		valid_kp_y_out.append(kp_y_out)

	X_valid = np.array(X_valid)
	y_valid_in = np.array(valid_kp_y_in)
	y_valid_out = np.array(valid_kp_y_out)

	x_in_connector = DataConnector(data_path, 'X_valid.pkl', X_valid)
	x_in_connector.save_pickle()
	y_in_connector = DataConnector(data_path, 'y_valid_in.pkl', y_valid_in)
	y_in_connector.save_pickle()
	y_out_connector = DataConnector(data_path, 'y_valid_out.pkl', y_valid_out)
	y_out_connector.save_pickle()



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

	test_in_tokens_connector = DataConnector(data_path, 'test_input_tokens.pkl', data=None)
	test_in_tokens_connector.read_pickle()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_tokens.pkl', data=None)
	test_out_tokens_connector.read_pickle()
	test_out_tokens = test_out_tokens_connector.read_file

	# create X,y pair for model training and evaluation
	X_test = np.zeros((len(test_in_tokens), encoder_length), dtype=np.int32)
	test_kp_y_in = []
	test_kp_y_out = []

	for i, (in_text, out_text) in enumerate(zip(test_in_tokens, test_out_tokens)):

		len_in_text = len(in_text)
		if len_in_text > encoder_length:
			txt = in_text[:encoder_length]
		else:
		  txt = in_text

		for t, word in enumerate(txt):
			if word in indices_words.values():
				X_test[i, t] = words_indices[word]
			# OOV (unknown words)
			else: 
				X_test[i, t] = words_indices['<unk>']

		kp_y_in = []
		kp_y_out = []

		for kp in keys:

			y_in = np.zeros((1, decoder_length+1), dtype=np.int32) 
			y_out = np.zeros((1, decoder_length+1), dtype=np.int32) 

			len_kp = len(kp)

			if len_kp > decoder_length:
				txt_kp = kp[:decoder_length]
			else:
				txt_kp = kp

			txt_in = list(txt_kp)
			txt_out = list(txt_kp)
			txt_in.insert(0,'<start>')
			txt_out.append('<end>')

			for k, word in enumerate(txt_in):
				if word in indices_words.values():
					y_in[0, k] = words_indices[word]
				else:
					y_in[0, k] = words_indices['<unk>']

			kp_y_in.append(y_in)

			for k, word in enumerate(txt_out):
				if word in indices_words.values():
					y_out[0, k] = words_indices[word]
				else:
					y_out[0, k] = words_indices['<unk>']

			kp_y_out.append(y_out)

		test_kp_y_in.append(kp_y_in)
		test_kp_y_out.append(kp_y_out)

	X_test = np.array(X_test)
	y_test_in = np.array(test_kp_y_in)
	y_test_out = np.array(test_kp_y_out)

	x_in_connector = DataConnector(data_path, 'X_test.pkl', X_test)
	x_in_connector.save_pickle()
	y_in_connector = DataConnector(data_path, 'y_test_in.pkl', y_test_in)
	y_in_connector.save_pickle()
	y_out_connector = DataConnector(data_path, 'y_test_out.pkl', y_test_out)
	y_out_connector.save_pickle()

def pair_train(params):

	data_path = params['data_path']

	'''
	read training set

	'''
	x_train_connector = DataConnector(data_path, 'X_train.pkl', data=None)
	x_train_connector.read_pickle()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_in.pkl', data=None)
	y_train_in_connector.read_pickle()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_out.pkl', data=None)
	y_train_out_connector.read_pickle()
	y_train_out = y_train_out_connector.read_file

	docid_pair_train = []
	x_pair_train = []
	y_pair_train_in = []
	y_pair_train_out = []

	for i, (y_in, y_out) in enumerate(zip(y_train_in, y_train_out)):
		for j in range(len(y_in)):
			docid_pair_train.append(i)
			x_pair_train.append(X_train[i])
			y_pair_train_in.append(y_in[j])
			y_pair_train_out.append(y_out[j])

	x_pair_train = np.array(x_pair_train)
	y_pair_train_in = np.array(y_pair_train_in)
	y_pair_train_out = np.array(y_pair_train_out)

	x_in_connector = DataConnector(data_path, 'x_pair_train.pkl', x_pair_train)
	x_in_connector.save_pickle()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in.pkl', y_pair_train_in)
	y_in_connector.save_pickle()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out.pkl', y_pair_train_out)
	y_out_connector.save_pickle()

def pair_valid(params):

	data_path = params['data_path']

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(data_path, 'X_valid.pkl', data=None)
	x_valid_connector.read_pickle()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(data_path, 'y_valid_in.pkl', data=None)
	y_valid_in_connector.read_pickle()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(data_path, 'y_valid_out.pkl', data=None)
	y_valid_out_connector.read_pickle()
	y_valid_out = y_valid_out_connector.read_file

	docid_pair_valid = []
	x_pair_valid = []
	y_pair_valid_in = []
	y_pair_valid_out = []

	for i, (y_in, y_out) in enumerate(zip(y_valid_in, y_valid_out)):
		for j in range(len(y_in)):
			docid_pair_valid.append(i)
			x_pair_valid.append(X_valid[i])
			y_pair_valid_in.append(y_in[j])
			y_pair_valid_out.append(y_out[j])

	x_pair_valid = np.array(x_pair_valid)
	y_pair_valid_in = np.array(y_pair_valid_in)
	y_pair_valid_out = np.array(y_pair_valid_out)

	x_in_connector = DataConnector(data_path, 'x_pair_valid.pkl', x_pair_valid)
	x_in_connector.save_pickle()
	y_in_connector = DataConnector(data_path, 'y_pair_valid_in.pkl', y_pair_valid_in)
	y_in_connector.save_pickle()
	y_out_connector = DataConnector(data_path, 'y_pair_valid_out.pkl', y_pair_valid_out)
	y_out_connector.save_pickle()
