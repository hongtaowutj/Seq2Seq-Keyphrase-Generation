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
import gensim

from utils.data_preprocessing_v2 import Preprocessing
from utils.indexing import Indexing
from utils.data_connector import DataConnector
from utils.sequences_processing import SequenceProcessing
from utils.true_keyphrases import TrueKeyphrases
from utils.pretrained_embedding import PrepareEmbedding

'''
For KP20K data set
'''
def preprocessing_train(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing training data...")
	sys.stdout.flush()

	training_data = []
	for line in open(os.path.join(data_path,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	train_in_text = []
	train_out_keyphrases = []

	for train_data in training_data:

		text = train_data['title'] + " . " + train_data['abstract']
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
	
	t1 = time.time()
	print("Preprocessing training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()



def preprocessing_sent_train(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing sent of training data...")
	sys.stdout.flush()

	training_data = []
	for line in open(os.path.join(data_path,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	train_in_text = []
	train_out_keyphrases = []

	for train_data in training_data:

		text = train_data['title'] + " . " + train_data['abstract']
		train_in_text.append(text)
		train_out_keyphrases.append(train_data['keyword'].split(';'))

	
	train_prep = Preprocessing()
	prep_inputs = train_prep.split_sent(train_in_text)
	prep_outputs = train_prep.preprocess_out(train_out_keyphrases)
	train_input_tokens = train_prep.tokenized_sent(prep_inputs)
	train_output_tokens = train_prep.tokenize_out(prep_outputs)

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', train_input_tokens)
	train_in_connector.save_numpys()
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', train_output_tokens)
	train_out_connector.save_numpys()
	
	t1 = time.time()
	print("Preprocessing sent of training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
For KP20K data set
'''

def preprocessing_valid(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing validation set ...")
	sys.stdout.flush()


	validation_data = []
	for line in open(os.path.join(data_path,'kp20k_validation.json'), 'r'):
		validation_data.append(json.loads(line))


	valid_in_text = []
	valid_out_keyphrases = []

	for valid_data in validation_data:

		text = valid_data['title'] + " . " + valid_data['abstract']
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

	t1 = time.time()
	print("Preprocessing validation set is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_sent_valid(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing sent of validation set ...")
	sys.stdout.flush()


	validation_data = []
	for line in open(os.path.join(data_path,'kp20k_validation.json'), 'r'):
		validation_data.append(json.loads(line))


	valid_in_text = []
	valid_out_keyphrases = []

	for valid_data in validation_data:

		text = valid_data['title'] + " . " + valid_data['abstract']
		valid_in_text.append(text)
		valid_out_keyphrases.append(valid_data['keyword'].split(';'))

	'''
	val_in_connector = DataConnector(data_path, 'raw_val_in.npy', valid_in_text)
	val_in_connector.save_numpys()
	val_out_connector = DataConnector(data_path, 'raw_val_out.npy', valid_out_keyphrases)
	val_out_connector.save_numpys()
	'''

	valid_prep = Preprocessing()
	prep_inputs = valid_prep.split_sent(valid_in_text)
	prep_outputs = valid_prep.preprocess_out(valid_out_keyphrases)
	valid_input_tokens = valid_prep.tokenized_sent(prep_inputs)
	valid_output_tokens = valid_prep.tokenize_out(prep_outputs)

	valid_in_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', valid_input_tokens)
	valid_in_connector.save_numpys()
	valid_out_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', valid_output_tokens)
	valid_out_connector.save_numpys()
	

	t1 = time.time()
	print("Preprocessing sent of validation set is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
For KP20K data set
'''
def preprocessing_test(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Reading raw test data...")
	sys.stdout.flush()

	testing_data = []
	for line in open(os.path.join(data_path,'kp20k_testing.json'), 'r'):
		testing_data.append(json.loads(line))

	test_in_text = []
	test_out_keyphrases = []

	for test_data in testing_data:

		text = test_data['title'] + " . " + test_data['abstract']
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

	t1 = time.time()
	print("Preprocessing test set is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_sent_test(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing sent of test data...")
	sys.stdout.flush()

	testing_data = []
	for line in open(os.path.join(data_path,'kp20k_testing.json'), 'r'):
		testing_data.append(json.loads(line))

	test_in_text = []
	test_out_keyphrases = []

	for test_data in testing_data:

		text = test_data['title'] + " . " + test_data['abstract']
		test_in_text.append(text)
		test_out_keyphrases.append(test_data['keyword'].split(';'))

	test_in_connector = DataConnector(data_path, 'raw_test_in.npy', test_in_text)
	test_in_connector.save_numpys()
	test_out_connector = DataConnector(data_path, 'raw_test_out.npy', test_out_keyphrases)
	test_out_connector.save_numpys()

	test_prep = Preprocessing()
	prep_inputs = test_prep.split_sent(test_in_text)
	prep_outputs = test_prep.preprocess_out(test_out_keyphrases)
	test_input_tokens = test_prep.tokenized_sent(prep_inputs)
	test_output_tokens = test_prep.tokenize_out(prep_outputs)

	test_in_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', test_input_tokens)
	test_in_connector.save_numpys()
	test_out_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', test_output_tokens)
	test_out_connector.save_numpys()

	t1 = time.time()
	print("Preprocessing sent of test set is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()
	

def compute_stats(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Computing statistics...")
	sys.stdout.flush()

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file	

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	

	sent_in = np.concatenate((train_in_tokens, valid_in_tokens, test_in_tokens))

	len_sent_x = []
	num_sent_x = []
	ids = []
	for i, doc in enumerate(sent_in):
		if (len(doc) < 100):
			num_sent_x.append(len(doc))
		for j, sent in enumerate(doc):
			if (len(sent)) < 50:
				ids.append((i,j))
				len_sent_x.append(len(sent))

	avg_num_sent = np.mean(np.array(num_sent_x))
	std_num_sent = np.std(np.array(num_sent_x))
	max_num_sent = max(num_sent_x)
	idx_num_sent = np.argmax(np.array(num_sent_x))

	avg_len_sent = np.mean(np.array(len_sent_x))
	std_len_sent = np.std(np.array(len_sent_x))
	max_len_sent = max(len_sent_x)
	idx_len_sent = np.argmax(np.array(len_sent_x))
	id_doc = ids[idx_len_sent]

	print("len_sent_x[:10]: %s"%(len_sent_x[:10]))
	print("num_sent_x[:10]: %s"%(num_sent_x[:10]))

	print("average number of sentences per document: %s"%avg_num_sent)
	print("standard deviation number of sentences per document: %s"%std_num_sent)
	print("max number of sentences per document: %s"%max_num_sent)
	print("Document with max number of sentences per document: %s"%(sent_in[idx_num_sent]))

	print("average number of words per sentences: %s"%avg_len_sent)
	print("standard deviation number of words per sentences: %s"%std_len_sent)
	print("max number of words per sentences: %s"%max_len_sent)
	print("Document with max length of sentences per document: %s"%(sent_in[id_doc[0]]))

	t1 = time.time()
	print("Computing statistics is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def indexing_data(params):

	data_path = params['data_path']

	t0 = time.time()
	print("Indexing...")
	sys.stdout.flush()

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


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)

	term_freq_conn = DataConnector(data_path, 'term_freq.pkl', term_freq)
	term_freq_conn.save_pickle()
	indices_words_conn = DataConnector(data_path, 'indices_words.pkl', indices_words)
	indices_words_conn.save_pickle()
	words_indices_conn = DataConnector(data_path, 'words_indices.pkl', words_indices)
	words_indices_conn.save_pickle()

	print("vocabulary size: %s"%len(indices_words))

	t1 = time.time()
	print("Indexing is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def indexing_sub_data(params):

	data_path = params['data_path']

	'''
	read all tokens from training, validation, and testing set
	to create vocabulary index of word tokens
	'''

	

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_input_tokens = train_in_connector.read_file
	train_input_tokens = train_input_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_output_tokens = train_out_connector.read_file
	train_output_tokens = train_output_tokens[:100000]

	# only use first 100,000 examples
	train_prep = Preprocessing()
	train_tokens = train_prep.get_all_tokens(train_input_tokens, train_output_tokens)
	

	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.npy', data=None)
	valid_tokens_connector.read_numpys()
	valid_tokens = valid_tokens_connector.read_file

	test_tokens_connector = DataConnector(data_path, 'test_tokens.npy', data=None)
	test_tokens_connector.read_numpys()
	test_tokens = test_tokens_connector.read_file

	#all_tokens = train_tokens + valid_tokens + test_tokens
	all_tokens = np.concatenate((train_tokens, valid_tokens, test_tokens))


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)
	indexing.save_(term_freq, 'term_freq_r2.pkl', data_path)
	indexing.save_(indices_words, 'indices_words_r2.pkl', data_path)
	indexing.save_(words_indices, 'words_indices_r2.pkl', data_path)
	
	print("vocabulary size: %s"%len(indices_words))

def indexing_sent_sub(params):

	data_path = params['data_path']

	'''
	read all tokens from training, validation, and testing set
	to create vocabulary index of word tokens
	'''

	

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_input_tokens = train_in_connector.read_file
	train_input_tokens = train_input_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_output_tokens = train_out_connector.read_file
	train_output_tokens = train_output_tokens[:100000]

	# only use first 100,000 examples
	train_prep = Preprocessing()
	train_tokens = train_prep.get_all_sent_tokens(train_input_tokens, train_output_tokens)


	valid_in_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_input_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_output_tokens = valid_out_connector.read_file

	valid_prep = Preprocessing()
	valid_tokens = valid_prep.get_all_sent_tokens(valid_input_tokens, valid_output_tokens)


	test_in_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_connector.read_numpys()
	test_input_tokens = test_in_connector.read_file
	test_out_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_connector.read_numpys()
	test_output_tokens = test_out_connector.read_file

	test_prep = Preprocessing()
	test_tokens = test_prep.get_all_sent_tokens(test_input_tokens, test_output_tokens)
	

	all_tokens = np.concatenate((train_tokens, valid_tokens, test_tokens))


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)
	indexing.save_(term_freq, 'term_freq_sent_r2.pkl', data_path)
	indexing.save_(indices_words, 'indices_words_sent_r2.pkl', data_path)
	indexing.save_(words_indices, 'words_indices_sent_r2.pkl', data_path)
	
	print("vocabulary size: %s"%len(indices_words))

'''
with merged-vocabulary dictionary
'''
def transform_train(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(data_path, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(data_path, 'X_train_pad.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(train_in_tokens, max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(data_path, 'X_train_sent_r3.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(data_path, 'X_train_pad_sent_r3.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in_sent_r3.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out_sent_r3.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_train_sent_v1(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(train_in_tokens, max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_sent.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_sent_v2(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(train_in_tokens, max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_sent.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_train_sent_fsoftmax_v1(params):

	data_path = params['data_path']
	max_sents = params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(train_in_tokens, max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_sent_fsoftmax.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_sent_fsoftmax_v2(params):

	data_path = params['data_path']
	max_sents = params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent-input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_sent-output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(train_in_tokens, max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_sent_fsoftmax.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_r2.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices_r2.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_fsoftmax_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_fsoftmax.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_fsoftmax_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming training set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_in_tokens = train_in_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file
	train_out_tokens = train_out_tokens[:100000]

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)


	print("\nnumber of examples in training set: %s\n"%(len(X_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(y_train_in)))
	sys.stdout.flush()

	x_in_connector = DataConnector(preprocessed_data, 'X_train_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_train_pad_fsoftmax.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(data_path, 'X_valid.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_valid_pad.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(data_path, 'X_valid_sent_r3.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_valid_pad_sent_r3.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_in_sent_r3.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_out_sent_r3.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_valid_sent_v1(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_valid_sent_v2(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_sent_fsoftmax_v1(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_sent_fsoftmax_v2(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_tokens_connector = DataConnector(data_path, 'valid_sent_input_tokens.npy', data=None)
	valid_in_tokens_connector.read_numpys()
	valid_in_tokens = valid_in_tokens_connector.read_file
	valid_out_tokens_connector = DataConnector(data_path, 'valid_sent_output_tokens.npy', data=None)
	valid_out_tokens_connector.read_numpys()
	valid_out_tokens = valid_out_tokens_connector.read_file

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_r2.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'all_words_indices_r2.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_fsoftmax_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_fsoftmax.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_valid_fsoftmax_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_fsoftmax.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_valid_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming validation set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices.pkl', data=None)
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

	print("\nnumber of validation examples in preprocessed data inputs: %s\n"%(len(valid_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of validation examples in preprocessed data outputs: %s\n"%(len(valid_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	
	print("X_valid_pad: %s"%str(X_valid_pad.shape))
	print("\nnumber of X examples in validation set: %s\n"%(len(X_valid)))
	sys.stdout.flush()
	print("\nnumber of Y examples in validation set: %s\n"%(len(y_valid_in)))
	sys.stdout.flush()
	
	x_in_connector = DataConnector(preprocessed_data, 'X_valid.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_test(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file


	reversed_vocab = DataConnector(data_path, 'all_words_indices.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_test.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_test_pad.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file


	reversed_vocab = DataConnector(data_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_test_sent_r3.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_test_pad_sent_r3.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_in_sent_r3.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_out_sent_r3.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_sent_v1(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()



def transform_test_sent_v2(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_test_sent_fsoftmax_v1(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_sent_fsoftmax_v2(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_test_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(data_path, 'all_indices_words_r2.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file


	reversed_vocab = DataConnector(data_path, 'all_words_indices_r2.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_fsoftmax_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_fsoftmax.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_test_fsoftmax_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_fsoftmax.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_v2(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_data, 'all_wordidx_vocabulary.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices.pkl', data=None)
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

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(test_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(test_out_tokens)))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_test.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_train(params):

	preprocessed_data = params['preprocessed_data']

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(preprocessed_data, 'X_train.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_train.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_train.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_train_sent_fsoftmax(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(preprocessed_data, 'X_train_sent_fsoftmax.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent_fsoftmax.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent_fsoftmax.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))


	x_pair_train = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_train_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_train_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_train_sent_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_train_sent_fsoftmax.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in_sent_fsoftmax.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out_sent_fsoftmax.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_train_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(preprocessed_data, 'X_train_sent.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in_sent.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out_sent.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))


	x_pair_train = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_train_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_train_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_train_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_train_sent.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in_sent.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out_sent.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_train_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(preprocessed_data, 'X_train_fsoftmax.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in_fsoftmax.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out_fsoftmax.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_train_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_train_fsoftmax.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in_fsoftmax.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out_fsoftmax.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_train_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(preprocessed_data, 'X_train.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_train.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_train.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_train_in.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_train_out.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_valid(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing validation set...")
	sys.stdout.flush()

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

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	x_pair_valid, y_pair_valid_in, y_pair_valid_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

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

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_valid_sent_fsoftmax(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing validation set...")
	sys.stdout.flush()

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(preprocessed_data, 'X_valid_sent_fsoftmax.npy', data=None)
	x_valid_connector.read_numpys()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent_fsoftmax.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent_fsoftmax.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	

	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))


	x_pair_valid = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_valid_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_valid_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_valid_sent_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_valid_sent_fsoftmax.npy', x_pair_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_valid_in_sent_fsoftmax.npy', y_pair_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_valid_out_sent_fsoftmax.npy', y_pair_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_valid_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing validation set...")
	sys.stdout.flush()

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(preprocessed_data, 'X_valid_sent.npy', data=None)
	x_valid_connector.read_numpys()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(preprocessed_data, 'y_valid_in_sent.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(preprocessed_data, 'y_valid_out_sent.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	

	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))


	x_pair_valid = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_valid_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_valid_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_valid_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_valid_sent.npy', x_pair_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_valid_in_sent.npy', y_pair_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_valid_out_sent.npy', y_pair_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_valid_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing validation set...")
	sys.stdout.flush()

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(preprocessed_data, 'X_valid_fsoftmax.npy', data=None)
	x_valid_connector.read_numpys()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(preprocessed_data, 'y_valid_in_fsoftmax.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(preprocessed_data, 'y_valid_out_fsoftmax.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	x_pair_valid, y_pair_valid_in, y_pair_valid_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair_valid: %s\n"%str(x_pair_valid.shape))
	print("\nshape of y_pair_valid_in: %s\n"%str(y_pair_valid_in.shape))
	print("\nshape of y_pair_valid_out: %s\n"%str(y_pair_valid_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_validpair_valid_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_valid_fsoftmax.npy', x_pair_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_valid_in_fsoftmax.npy', y_pair_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_valid_out_fsoftmax.npy', y_pair_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_valid_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing validation set...")
	sys.stdout.flush()

	'''
	read validation set

	'''
	x_valid_connector = DataConnector(preprocessed_data, 'X_valid.npy', data=None)
	x_valid_connector.read_numpys()
	X_valid = x_valid_connector.read_file

	y_valid_in_connector = DataConnector(preprocessed_data, 'y_valid_in.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(preprocessed_data, 'y_valid_out.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	x_pair_valid, y_pair_valid_in, y_pair_valid_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair_valid: %s\n"%str(x_pair_valid.shape))
	print("\nshape of y_pair_valid_in: %s\n"%str(y_pair_valid_in.shape))
	print("\nshape of y_pair_valid_out: %s\n"%str(y_pair_valid_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_valid.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_valid.npy', x_pair_valid)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_valid_in.npy', y_pair_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_valid_out.npy', y_pair_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_test(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read test set

	'''
	X_test_connector = DataConnector(data_path, 'X_test.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_in.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_out.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n X_test original shape before being paired: %s\n"%len(X_test))
	print("\n y_test_in original shape before : %s\n"%len(y_test_in))
	print("\n y_test_out original shape before : %s\n"%len(y_test_out))

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


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

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_test_sent_fsoftmax(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read test set

	'''
	X_test_connector = DataConnector(preprocessed_data, 'X_test_sent_fsoftmax.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent_fsoftmax.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent_fsoftmax.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n X_test original shape before being paired: %s\n"%len(X_test))
	print("\n y_test_in original shape before : %s\n"%len(y_test_in))
	print("\n y_test_out original shape before : %s\n"%len(y_test_out))

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)


	x_pair_test = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_test_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_test_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_test_sent_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_test_sent_fsoftmax.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_test_in_sent_fsoftmax.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_test_out_sent_fsoftmax.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_test_sent_sub(params):

	data_path = params['data_path']
	max_sents= params['max_sents']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read test set

	'''
	X_test_connector = DataConnector(preprocessed_data, 'X_test_sent.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in_sent.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out_sent.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n X_test original shape before being paired: %s\n"%len(X_test))
	print("\n y_test_in original shape before : %s\n"%len(y_test_in))
	print("\n y_test_out original shape before : %s\n"%len(y_test_out))

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)


	x_pair_test = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair)
	y_pair_test_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_in)
	y_pair_test_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_test_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_test_sent.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_test_in_sent.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_test_out_sent.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_test_sub(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read test set

	'''
	X_test_connector = DataConnector(preprocessed_data, 'X_test.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n X_test original shape before being paired: %s\n"%len(X_test))
	print("\n y_test_in original shape before : %s\n"%len(y_test_in))
	print("\n y_test_out original shape before : %s\n"%len(y_test_out))

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_test.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_test.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_test_in.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_test_out.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_test_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Transforming test set into integer sequences...")
	sys.stdout.flush()

	'''
	read test set

	'''
	X_test_connector = DataConnector(preprocessed_data, 'X_test_fsoftmax.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in_fsoftmax.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out_fsoftmax.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n X_test original shape before being paired: %s\n"%len(X_test))
	print("\n y_test_in original shape before : %s\n"%len(y_test_in))
	print("\n y_test_out original shape before : %s\n"%len(y_test_out))

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_test_fsoftmax.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_test_fsoftmax.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_test_in_fsoftmax.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_test_out_fsoftmax.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
Get average number of key phrases per document in corpus
'''

def compute_average_keyphrases(params):

	data_path = params['data_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing average key phrases per document...")
	sys.stdout.flush()

	# from training set
	train_kp_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	# from validation set
	valid_kp_connector = DataConnector(data_path, 'valid_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	# from test set

	test_kp_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	all_keyphrases = np.concatenate((train_kps, valid_kps, test_kps))


	# transform tokenized y_true (ground truth of keyphrases) into full sentences / keyphrases
	keyphrases_transform =  TrueKeyphrases(all_keyphrases)
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


	t1 = time.time()
	print("Computing average key phrases done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
compute max, average length of key phrases
'''

def compute_keyphrase_length(params):

	data_path = params['data_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing statistics of key phrases per document...")
	sys.stdout.flush()

	# from training set
	train_kp_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	# from validation set
	valid_kp_connector = DataConnector(data_path, 'valid_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	# from test set

	test_kp_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	all_keyphrases = np.concatenate((train_kps, valid_kps, test_kps))

	len_kps = []
	for i, kp_list in enumerate(all_keyphrases):
		for j, kp in enumerate(kp_list):
			len_kps.append(len(kp))
			if len(kp) > 20:
				print("i,j: (%s, %s)"%(i,j))
				print("kp: %s"%kp)

	max_kps = max(len_kps)
	mean_kps = np.mean(np.array(len_kps))
	std_kps = np.std(np.array(len_kps))


	print("Maximum number of words per key phrase: %s" %max_kps)
	sys.stdout.flush()
	print("Average number of words per key phrase: %s" %mean_kps)
	sys.stdout.flush()
	print("Standard Deviation of number of words per key phrase: %s" %std_kps)
	sys.stdout.flush()


	t1 = time.time()
	print("Computing statistics of key phrases done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def compute_presence_absence(params):

	data_path = params['data_path']
	

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing presence or absence of key phrases per document...")
	sys.stdout.flush()

	# from training set

	training_data = []
	for line in open(os.path.join(data_path,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	train_in_text = []
	train_out_keyphrases = []

	for train_data in training_data:

		text = train_data['title'] + " . " + train_data['abstract']
		train_in_text.append(text)
		train_out_keyphrases.append(train_data['keyword'].split(';'))

	print("train_out_keyphrases[6479]: %s"%train_out_keyphrases[6479])

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file

	train_kp_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	print("train_kps[6479]: %s"%train_kps[6479])

	compute_presence_train = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	all_npresence_train, all_nabsence_train = compute_presence_train.compute_presence(train_in_tokens, train_kps)
	total_train = np.sum(all_npresence_train) + np.sum(all_nabsence_train)


	# from validation set

	valid_in_connector = DataConnector(data_path, 'valid_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file

	valid_kp_connector = DataConnector(data_path, 'valid_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	compute_presence_val = SequenceProcessing()
	all_npresence_val, all_nabsence_val = compute_presence_val.compute_presence(valid_in_tokens, valid_kps)
	total_val = np.sum(all_npresence_val) + np.sum(all_nabsence_val)

	# from test set

	test_in_tokens_connector = DataConnector(data_path, 'test_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file

	test_kp_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	compute_presence_test = SequenceProcessing()
	all_npresence_test, all_nabsence_test = compute_presence_test.compute_presence(test_in_tokens, test_kps)
	total_test = np.sum(all_npresence_test) + np.sum(all_nabsence_test)

	n_presence = np.sum(all_npresence_train) + np.sum(all_npresence_val) + np.sum(all_npresence_test)
	n_absence = np.sum(all_nabsence_train) + np.sum(all_nabsence_val) + np.sum(all_nabsence_test)
	total = total_train + total_val +  total_test

	persen_absence = n_absence / total
	persen_presence = n_presence / total

	print(" Absent key phrase: %s" %persen_absence)
	sys.stdout.flush()
	print(" Present key phrase: %s" %persen_presence)
	sys.stdout.flush()


	t1 = time.time()
	print("Computing presence or absence of key phrases per document done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
merge vocabularies of all data set
'''

def merge_tfs_fsoftmax(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']

	# KP20k corpus

	tf_kp20k_conn = DataConnector(data_path, 'term_freq_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file

	print("length tf_kp20k: %s"%len(tf_kp20k))
	sys.stdout.flush()

	top_20_tf_kp20k = sorted(tf_kp20k.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_kp20k:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_kp20k:
		print(k, v)
		sys.stdout.flush()

	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

	print("length tf_inspec: %s"%len(tf_inspec))
	sys.stdout.flush()

	top_20_tf_inspec = sorted(tf_inspec.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_inspec:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_inspec:
		print(k, v)
		sys.stdout.flush()

	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	print("length tf_krapivin: %s"%len(tf_krapivin))
	sys.stdout.flush()

	top_20_tf_krapivin = sorted(tf_krapivin.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_krapivin:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_krapivin:
		print(k, v)
		sys.stdout.flush()

	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	print("length tf_nus: %s"%len(tf_nus))
	sys.stdout.flush()

	top_20_tf_nus = sorted(tf_nus.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_nus:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_nus:
		print(k, v)
		sys.stdout.flush()

	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	print("length tf_semeval: %s"%len(tf_semeval))
	sys.stdout.flush()


	top_20_tf_semeval = sorted(tf_semeval.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_semeval:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_semeval:
		print(k, v)
		sys.stdout.flush()

	all_tfs = {}

	all_tfs.update(tf_kp20k)

	for (k,v) in tf_inspec.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_krapivin.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_nus.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_semeval.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)


	print("length all_tfs: %s"%len(all_tfs))
	sys.stdout.flush()

	print("all_tfs: %s"%str(list(all_tfs.items())[:10]))
	sys.stdout.flush()

	top_20 = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in all_tfs:")
	sys.stdout.flush()

	for (k,v) in top_20:
		print(k, v)
		sys.stdout.flush()

	# Get 10K-top most frequent words from combined vocabularies:
	#top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:10000]
	top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:10000]
	top_words={}
	top_words.update(top_freq)

	print("top_words and frequency: %s"%str(list(top_words.items())[:20]))

	vocab_all = list(top_words.keys())
	print("length all vocab: %s"%len(vocab_all))
	print("top 10-vocab: %s"%str(vocab_all[:10]))

	## indexing from new constructed word /vocab list

	vocab_all.insert(0,'<pad>')
	vocab_all.append('<start>')
	vocab_all.append('<end>')
	vocab_all.append('<unk>')

	vocab=dict([(i,vocab_all[i]) for i in range(len(vocab_all))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(preprocessed_data, 'all_indices_words_fsoftmax.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(preprocessed_data, 'all_words_indices_fsoftmax.pkl', words_indices)
	words_indices_conn.save_pickle()


	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def merge_tfs(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']

	# KP20k corpus

	tf_kp20k_conn = DataConnector(data_path, 'term_freq_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file

	print("length tf_kp20k: %s"%len(tf_kp20k))
	sys.stdout.flush()

	top_20_tf_kp20k = sorted(tf_kp20k.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_kp20k:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_kp20k:
		print(k, v)
		sys.stdout.flush()

	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

	print("length tf_inspec: %s"%len(tf_inspec))
	sys.stdout.flush()

	top_20_tf_inspec = sorted(tf_inspec.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_inspec:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_inspec:
		print(k, v)
		sys.stdout.flush()

	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	print("length tf_krapivin: %s"%len(tf_krapivin))
	sys.stdout.flush()

	top_20_tf_krapivin = sorted(tf_krapivin.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_krapivin:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_krapivin:
		print(k, v)
		sys.stdout.flush()

	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	print("length tf_nus: %s"%len(tf_nus))
	sys.stdout.flush()

	top_20_tf_nus = sorted(tf_nus.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_nus:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_nus:
		print(k, v)
		sys.stdout.flush()

	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	print("length tf_semeval: %s"%len(tf_semeval))
	sys.stdout.flush()


	top_20_tf_semeval = sorted(tf_semeval.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in tf_semeval:")
	sys.stdout.flush()

	for (k,v) in top_20_tf_semeval:
		print(k, v)
		sys.stdout.flush()

	all_tfs = {}

	all_tfs.update(tf_kp20k)

	for (k,v) in tf_inspec.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_krapivin.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_nus.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_semeval.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)


	print("length all_tfs: %s"%len(all_tfs))
	sys.stdout.flush()

	print("all_tfs: %s"%str(list(all_tfs.items())[:10]))
	sys.stdout.flush()

	top_20 = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:20]
	print("20-top words in all_tfs:")
	sys.stdout.flush()

	for (k,v) in top_20:
		print(k, v)
		sys.stdout.flush()

	# Get 10K-top most frequent words from combined vocabularies:
	#top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:10000]
	top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)
	top_words={}
	top_words.update(top_freq)

	print("top_words and frequency: %s"%str(list(top_words.items())[:20]))

	vocab_all = list(top_words.keys())
	print("length all vocab: %s"%len(vocab_all))
	print("top 10-vocab: %s"%str(vocab_all[:10]))

	## indexing from new constructed word /vocab list

	vocab_all.insert(0,'<pad>')
	vocab_all.append('<start>')
	vocab_all.append('<end>')
	vocab_all.append('<unk>')

	vocab=dict([(i,vocab_all[i]) for i in range(len(vocab_all))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(preprocessed_data, 'all_indices_words.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(preprocessed_data, 'all_words_indices.pkl', words_indices)
	words_indices_conn.save_pickle()


	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def merge_tfs_sent(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']
	

	# KP20k corpus

	tf_kp20k_conn = DataConnector(data_path, 'term_freq_sent_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file


	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

		
	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	
	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	
	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	all_tfs = {}

	all_tfs.update(tf_kp20k)

	for (k,v) in tf_inspec.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_krapivin.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_nus.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_semeval.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)


	print("length all_tfs: %s"%len(all_tfs))
	sys.stdout.flush()

	# Get 10K-top most frequent words from combined vocabularies:
	top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)
	top_words={}
	top_words.update(top_freq)

	print("top_words and frequency: %s"%str(list(top_words.items())[:20]))

	vocab_all = list(top_words.keys())
	print("length all vocab: %s"%len(vocab_all))
	print("top 10-vocab: %s"%str(vocab_all[:10]))

	## indexing from new constructed word /vocab list

	vocab_all.insert(0,'<pad>')
	vocab_all.append('<start>')
	vocab_all.append('<end>')
	vocab_all.append('<unk>')

	vocab=dict([(i,vocab_all[i]) for i in range(len(vocab_all))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(preprocessed_data, 'all_indices_words_sent.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(preprocessed_data, 'all_words_indices_sent.pkl', words_indices)
	words_indices_conn.save_pickle()
	

	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def merge_tfs_sent_fsoftmax(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']
	

	# KP20k corpus

	tf_kp20k_conn = DataConnector(data_path, 'term_freq_sent_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file


	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

		
	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	
	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	
	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	all_tfs = {}

	all_tfs.update(tf_kp20k)

	for (k,v) in tf_inspec.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_krapivin.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_nus.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)

	for (k,v) in tf_semeval.items():
		if k not in all_tfs.keys():
			all_tfs[k] = int(v)
		else:
			all_tfs[k] += int(v)


	print("length all_tfs: %s"%len(all_tfs))
	sys.stdout.flush()

	# Get 10K-top most frequent words from combined vocabularies:
	top_freq = sorted(all_tfs.items(), key = lambda t: t[1], reverse = True)[:10000]
	top_words={}
	top_words.update(top_freq)

	print("top_words and frequency: %s"%str(list(top_words.items())[:20]))

	vocab_all = list(top_words.keys())
	print("length all vocab: %s"%len(vocab_all))
	print("top 10-vocab: %s"%str(vocab_all[:10]))

	## indexing from new constructed word /vocab list

	vocab_all.insert(0,'<pad>')
	vocab_all.append('<start>')
	vocab_all.append('<end>')
	vocab_all.append('<unk>')

	vocab=dict([(i,vocab_all[i]) for i in range(len(vocab_all))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(preprocessed_data, 'all_indices_words_sent_fsoftmax.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(preprocessed_data, 'all_words_indices_sent_fsoftmax.pkl', words_indices)
	words_indices_conn.save_pickle()
	

	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

	
def merge_vocab(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']
	

	# KP20k corpus

	tf_kp20k_conn = DataConnector(kp20k_path, 'term_freq_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file

	common_words = tf_kp20k.most_common(len(tf_kp20k))
	print("common_words [100:120] in kp20k: %s"%common_words[100:120])
	sys.stdout.flush()

	print("less frequent words in kp20k: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_kp20k = list(tf_kp20k.keys())


	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

	common_words = tf_inspec.most_common(len(tf_inspec))
	print("common_words [100:120] in inspec: %s"%common_words[100:120])

	print("less frequent words in Inspec: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_inspec = list(tf_inspec.keys())

	
	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	common_words = tf_krapivin.most_common(len(tf_krapivin))
	print("common_words [100:120] in krapivin: %s"%common_words[100:120])
	sys.stdout.flush()

	print("less frequent words in Krapivin: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_krapivin = list(tf_krapivin.keys())

	

	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	common_words = tf_nus.most_common(len(tf_nus))
	print("common_words [100:120] in NUS: %s"%common_words[100:120])

	print("less frequent words in NUS: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_nus = list(tf_nus.keys())	

	

	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	common_words = tf_semeval.most_common(len(tf_semeval))
	print("common_words [100:120] in semeval: %s"%common_words[100:120])

	print("less frequent words in SemEval2010: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_semeval = list(tf_semeval.keys())	



	## merging all vocab

	vocab_merge = list(set(np.concatenate((vocab_kp20k, vocab_inspec, vocab_krapivin, vocab_nus, vocab_semeval))))

	## indexing from new constructed word /vocab list

	vocab_merge.insert(0,'<pad>')
	vocab_merge.append('<start>')
	vocab_merge.append('<end>')
	vocab_merge.append('<unk>')

	vocab=dict([(i,vocab_merge[i]) for i in range(len(vocab_merge))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(kp20k_path, 'all_indices_words_r2.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(kp20k_path, 'all_words_indices_r2.pkl', words_indices)
	words_indices_conn.save_pickle()

	print("all_indices_words[:50]: %s"%list(indices_words.items())[:50])

	print("all_words_indices[:50]: %s"%list(words_indices.items())[:50])

	print("vocabulary size: %s"%len(indices_words))

	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def merge_vocab_sent(params):

	t0 = time.time()

	print("Merging vocab from all data sets...")
	sys.stdout.flush()

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	inspec_path = params['inspec_path']
	krapivin_path = params['krapivin_path']
	nus_path = params['nus_path']
	semeval_path = params['semeval_path']
	

	# KP20k corpus

	tf_kp20k_conn = DataConnector(kp20k_path, 'term_freq_sent_r2.pkl', data=None)
	tf_kp20k_conn.read_pickle()
	tf_kp20k = tf_kp20k_conn.read_file

	common_words = tf_kp20k.most_common(len(tf_kp20k))
	print("common_words [100:120] in kp20k: %s"%common_words[100:120])
	sys.stdout.flush()

	print("less frequent words in kp20k: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_kp20k = list(tf_kp20k.keys())


	# Inspec corpus

	tf_inspec_conn = DataConnector(inspec_path, 'term_freq.pkl', data=None)
	tf_inspec_conn.read_pickle()
	tf_inspec = tf_inspec_conn.read_file

	common_words = tf_inspec.most_common(len(tf_inspec))
	print("common_words [100:120] in inspec: %s"%common_words[100:120])

	print("less frequent words in Inspec: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_inspec = list(tf_inspec.keys())

	
	# Krapivin corpus

	tf_krapivin_conn = DataConnector(krapivin_path, 'term_freq.pkl', data=None)
	tf_krapivin_conn.read_pickle()
	tf_krapivin = tf_krapivin_conn.read_file

	common_words = tf_krapivin.most_common(len(tf_krapivin))
	print("common_words [100:120] in krapivin: %s"%common_words[100:120])
	sys.stdout.flush()

	print("less frequent words in Krapivin: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_krapivin = list(tf_krapivin.keys())

	

	# NUS corpus

	tf_nus_conn = DataConnector(nus_path, 'term_freq.pkl', data=None)
	tf_nus_conn.read_pickle()
	tf_nus = tf_nus_conn.read_file

	common_words = tf_nus.most_common(len(tf_nus))
	print("common_words [100:120] in NUS: %s"%common_words[100:120])

	print("less frequent words in NUS: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_nus = list(tf_nus.keys())	

	

	# SemEval2010 corpus

	tf_semeval_conn = DataConnector(semeval_path, 'term_freq.pkl', data=None)
	tf_semeval_conn.read_pickle()
	tf_semeval = tf_semeval_conn.read_file

	common_words = tf_semeval.most_common(len(tf_semeval))
	print("common_words [100:120] in semeval: %s"%common_words[100:120])

	print("less frequent words in SemEval2010: %s"%common_words[-50:len(common_words)])
	sys.stdout.flush()

	vocab_semeval = list(tf_semeval.keys())	

	

	## merging all vocab

	vocab_merge = list(set(np.concatenate((vocab_kp20k, vocab_inspec, vocab_krapivin, vocab_nus, vocab_semeval))))

	## indexing from new constructed word /vocab list

	vocab_merge.insert(0,'<pad>')
	vocab_merge.append('<start>')
	vocab_merge.append('<end>')
	vocab_merge.append('<unk>')

	vocab=dict([(i,vocab_merge[i]) for i in range(len(vocab_merge))])
	indices_words = vocab
	words_indices = dict((v,k) for (k,v) in indices_words.items())

	indices_words_conn = DataConnector(kp20k_path, 'all_indices_words_sent_r2.pkl', indices_words)
	indices_words_conn.save_pickle()

	words_indices_conn = DataConnector(kp20k_path, 'all_words_indices_sent_r2.pkl', words_indices)
	words_indices_conn.save_pickle()

	print("all_indices_words[:50]: %s"%list(indices_words.items())[:50])

	print("all_words_indices[:50]: %s"%list(words_indices.items())[:50])

	print("vocabulary size: %s"%len(indices_words))

	t1 = time.time()
	print("Merging vocab of 5 data sets done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def merge_order_vocab(params):

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_input_tokens = train_in_connector.read_file
	train_input_tokens = train_input_tokens[:100000]
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_output_tokens = train_out_connector.read_file
	train_output_tokens = train_output_tokens[:100000]

	# only use first 100,000 examples
	train_prep = Preprocessing()
	train_tokens = train_prep.get_all_tokens(train_input_tokens, train_output_tokens)
	

	valid_tokens_connector = DataConnector(data_path, 'valid_tokens.npy', data=None)
	valid_tokens_connector.read_numpys()
	valid_tokens = valid_tokens_connector.read_file

	test_tokens_connector = DataConnector(data_path, 'test_tokens.npy', data=None)
	test_tokens_connector.read_numpys()
	test_tokens = test_tokens_connector.read_file

	#all_tokens = train_tokens + valid_tokens + test_tokens
	all_tokens = np.concatenate((train_tokens, valid_tokens, test_tokens))


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)
	indexing.save_(term_freq, 'term_freq_r2.pkl', data_path)
	indexing.save_(indices_words, 'indices_words_r2.pkl', data_path)
	indexing.save_(words_indices, 'words_indices_r2.pkl', data_path)
	
	print("vocabulary size: %s"%len(indices_words))

def create_in_out_vocab_fsoftmax(params):

	t0 = time.time()

	print("Reindexing vocabulary with in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	preprocessed_data = params['preprocessed_data']

	#gensim.scripts.glove2word2vec.glove2word2vec(os.path.join(glove_path, glove_name), os.path.join(preprocessed_data, 'word2vec.glove.100d.txt'))

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data,'word2vec.glove.100d.txt'))

	print("Length of (number of words) in glove pretrained embedding: %s"%len(list(glove.vocab.keys())))
	sys.stdout.flush()

	id_word_conn = DataConnector(preprocessed_data, 'all_indices_words_fsoftmax.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	word_id_conn = DataConnector(preprocessed_data, 'all_words_indices_fsoftmax.pkl', data=None)
	word_id_conn.read_pickle()
	word_id = word_id_conn.read_file

	print("Vocabulary size in observed corpus (5 data): %s"%len(word_id))
	sys.stdout.flush()

	in_words = []
	out_words = []
	for k,v in id_word.items():
		if (k==0) | (v in list(glove.vocab.keys())):
			in_words.append(v)
		else:
			out_words.append(v)

	print("Size of in-domain vocabulary (valid words): %s"%len(in_words))
	sys.stdout.flush()
	print("Size of out-of-vocabulary: %s"%len(out_words))
	sys.stdout.flush()

	all_words = np.concatenate((in_words, out_words))
	in_id_word = dict([(i, in_words[i]) for i in range(len(in_words))])
	out_id_word = dict([(i, out_words[i]) for i in range(len(out_words))])
	all_id_word = dict([(i, all_words[i]) for i in range(len(all_words))])
	all_word_id = dict((v,k) for (k,v) in all_id_word.items())

	in_id_word_conn = DataConnector(preprocessed_data, 'in_vocabulary_fsoftmax.pkl', in_id_word)
	in_id_word_conn.save_pickle()

	out_id_word_conn = DataConnector(preprocessed_data, 'out_vocabulary_fsoftmax.pkl', out_id_word)
	out_id_word_conn.save_pickle()

	all_id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_fsoftmax.pkl', all_id_word)
	all_id_word_conn.save_pickle()

	all_word_id_conn = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_fsoftmax.pkl', all_word_id)
	all_word_id_conn.save_pickle()

	t1 = time.time()
	print("Reindexing is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def create_in_out_vocab(params):

	t0 = time.time()

	print("Reindexing vocabulary with in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	preprocessed_data = params['preprocessed_data']

	gensim.scripts.glove2word2vec.glove2word2vec(os.path.join(glove_path, glove_name), os.path.join(preprocessed_data, 'word2vec.glove.100d.txt'))

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data,'word2vec.glove.100d.txt'))

	print("Length of (number of words) in glove pretrained embedding: %s"%len(list(glove.vocab.keys())))
	sys.stdout.flush()

	id_word_conn = DataConnector(preprocessed_data, 'all_indices_words.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	word_id_conn = DataConnector(preprocessed_data, 'all_words_indices.pkl', data=None)
	word_id_conn.read_pickle()
	word_id = word_id_conn.read_file

	print("Vocabulary size in observed corpus (5 data): %s"%len(word_id))
	sys.stdout.flush()

	in_words = []
	out_words = []
	for k,v in id_word.items():
		if (k==0) | (v in list(glove.vocab.keys())):
			in_words.append(v)
		else:
			out_words.append(v)

	print("Size of in-domain vocabulary (valid words): %s"%len(in_words))
	sys.stdout.flush()
	print("Size of out-of-vocabulary: %s"%len(out_words))
	sys.stdout.flush()

	all_words = np.concatenate((in_words, out_words))
	in_id_word = dict([(i, in_words[i]) for i in range(len(in_words))])
	out_id_word = dict([(i, out_words[i]) for i in range(len(out_words))])
	all_id_word = dict([(i, all_words[i]) for i in range(len(all_words))])
	all_word_id = dict((v,k) for (k,v) in all_id_word.items())

	in_id_word_conn = DataConnector(preprocessed_data, 'in_vocabulary.pkl', in_id_word)
	in_id_word_conn.save_pickle()

	out_id_word_conn = DataConnector(preprocessed_data, 'out_vocabulary.pkl', out_id_word)
	out_id_word_conn.save_pickle()

	all_id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', all_id_word)
	all_id_word_conn.save_pickle()

	all_word_id_conn = DataConnector(preprocessed_data, 'all_wordidx_vocabulary.pkl', all_word_id)
	all_word_id_conn.save_pickle()

	t1 = time.time()
	print("Reindexing is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def create_in_out_vocab_sent(params):

	t0 = time.time()

	print("Reindexing vocabulary with in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	preprocessed_data = params['preprocessed_data']

	gensim.scripts.glove2word2vec.glove2word2vec(os.path.join(glove_path, glove_name), os.path.join(preprocessed_data, 'word2vec.glove.100d.txt'))

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data,'word2vec.glove.100d.txt'))

	print("Length of (number of words) in glove pretrained embedding: %s"%len(list(glove.vocab.keys())))
	sys.stdout.flush()

	id_word_conn = DataConnector(preprocessed_data, 'all_indices_words_sent.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	word_id_conn = DataConnector(preprocessed_data, 'all_words_indices_sent.pkl', data=None)
	word_id_conn.read_pickle()
	word_id = word_id_conn.read_file

	print("Vocabulary size in observed corpus (5 data): %s"%len(word_id))
	sys.stdout.flush()

	in_words = []
	out_words = []
	for k,v in id_word.items():
		if (k==0) | (v in list(glove.vocab.keys())):
			in_words.append(v)
		else:
			out_words.append(v)

	print("Size of in-domain vocabulary (valid words): %s"%len(in_words))
	sys.stdout.flush()
	print("Size of out-of-vocabulary: %s"%len(out_words))
	sys.stdout.flush()

	all_words = np.concatenate((in_words, out_words))
	in_id_word = dict([(i, in_words[i]) for i in range(len(in_words))])
	out_id_word = dict([(i, out_words[i]) for i in range(len(out_words))])
	all_id_word = dict([(i, all_words[i]) for i in range(len(all_words))])
	all_word_id = dict((v,k) for (k,v) in all_id_word.items())

	in_id_word_conn = DataConnector(preprocessed_data, 'in_vocabulary_sent.pkl', in_id_word)
	in_id_word_conn.save_pickle()

	out_id_word_conn = DataConnector(preprocessed_data, 'out_vocabulary_sent.pkl', out_id_word)
	out_id_word_conn.save_pickle()

	all_id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', all_id_word)
	all_id_word_conn.save_pickle()

	all_word_id_conn = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent.pkl', all_word_id)
	all_word_id_conn.save_pickle()

	t1 = time.time()
	print("Reindexing is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def create_in_out_vocab_sent_fsoftmax(params):

	t0 = time.time()

	print("Reindexing vocabulary with in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	preprocessed_data = params['preprocessed_data']

	gensim.scripts.glove2word2vec.glove2word2vec(os.path.join(glove_path, glove_name), os.path.join(preprocessed_data, 'word2vec.glove.100d.txt'))

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data,'word2vec.glove.100d.txt'))

	print("Length of (number of words) in glove pretrained embedding: %s"%len(list(glove.vocab.keys())))
	sys.stdout.flush()

	id_word_conn = DataConnector(preprocessed_data, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	word_id_conn = DataConnector(preprocessed_data, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	word_id_conn.read_pickle()
	word_id = word_id_conn.read_file

	print("Vocabulary size in observed corpus (5 data): %s"%len(word_id))
	sys.stdout.flush()

	in_words = []
	out_words = []
	for k,v in id_word.items():
		if (k==0) | (v in list(glove.vocab.keys())):
			in_words.append(v)
		else:
			out_words.append(v)

	print("Size of in-domain vocabulary (valid words): %s"%len(in_words))
	sys.stdout.flush()
	print("Size of out-of-vocabulary: %s"%len(out_words))
	sys.stdout.flush()

	all_words = np.concatenate((in_words, out_words))
	in_id_word = dict([(i, in_words[i]) for i in range(len(in_words))])
	out_id_word = dict([(i, out_words[i]) for i in range(len(out_words))])
	all_id_word = dict([(i, all_words[i]) for i in range(len(all_words))])
	all_word_id = dict((v,k) for (k,v) in all_id_word.items())

	in_id_word_conn = DataConnector(preprocessed_data, 'in_vocabulary_sent_fsoftmax.pkl', in_id_word)
	in_id_word_conn.save_pickle()

	out_id_word_conn = DataConnector(preprocessed_data, 'out_vocabulary_sent_fsoftmax.pkl', out_id_word)
	out_id_word_conn.save_pickle()

	all_id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent_fsoftmax.pkl', all_id_word)
	all_id_word_conn.save_pickle()

	all_word_id_conn = DataConnector(preprocessed_data, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', all_word_id)
	all_word_id_conn.save_pickle()

	t1 = time.time()
	print("Reindexing is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def create_embeddings_fsoftmax(params):

	t0 = time.time()

	print("Initializing embedding matrix of in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	glove_w2v = params['glove_w2v']
	preprocessed_data = params['preprocessed_data']
	embedding_dim = ['embedding_dim']

	id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data, glove_w2v))

	print("Initializing nontrainable embeddings...")
	sys.stdout.flush()

	matrix_glove = PrepareEmbedding()
	nontrainable_embed = matrix_glove.create_nontrainable(id_word, glove, 100)

	print("Initializing trainable embeddings...")
	sys.stdout.flush()

	matrix_oov = PrepareEmbedding()
	trainable_embed = matrix_oov.create_trainable(id_word, 100)

	pretrained_conn = DataConnector(preprocessed_data, 'nontrainable_embeddings_fsoftmax.pkl', nontrainable_embed)
	pretrained_conn.save_pickle()

	oov_conn = DataConnector(preprocessed_data, 'trainable_embeddings_fsoftmax.pkl', trainable_embed)
	oov_conn.save_pickle()

	t1 = time.time()
	print("Initialize embedding matrix is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def create_embeddings(params):

	t0 = time.time()

	print("Initializing embedding matrix of in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	glove_w2v = params['glove_w2v']
	preprocessed_data = params['preprocessed_data']
	embedding_dim = ['embedding_dim']

	id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data, glove_w2v))

	print("Initializing nontrainable embeddings...")
	sys.stdout.flush()

	matrix_glove = PrepareEmbedding()
	nontrainable_embed = matrix_glove.create_nontrainable(id_word, glove, 100)

	print("Initializing trainable embeddings...")
	sys.stdout.flush()

	matrix_oov = PrepareEmbedding()
	trainable_embed = matrix_oov.create_trainable(id_word, 100)

	pretrained_conn = DataConnector(preprocessed_data, 'nontrainable_embeddings.pkl', nontrainable_embed)
	pretrained_conn.save_pickle()

	oov_conn = DataConnector(preprocessed_data, 'trainable_embeddings.pkl', trainable_embed)
	oov_conn.save_pickle()

	t1 = time.time()
	print("Initialize embedding matrix is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()



def create_embeddings_sent(params):

	t0 = time.time()

	print("Initializing embedding matrix of in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	glove_w2v = params['glove_w2v']
	preprocessed_data = params['preprocessed_data']
	embedding_dim = ['embedding_dim']

	id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data, glove_w2v))

	print("Initializing nontrainable embeddings...")
	sys.stdout.flush()

	matrix_glove = PrepareEmbedding()
	nontrainable_embed = matrix_glove.create_nontrainable(id_word, glove, 100)

	print("Initializing trainable embeddings...")
	sys.stdout.flush()

	matrix_oov = PrepareEmbedding()
	trainable_embed = matrix_oov.create_trainable(id_word, 100)

	pretrained_conn = DataConnector(preprocessed_data, 'nontrainable_embeddings_sent.pkl', nontrainable_embed)
	pretrained_conn.save_pickle()

	oov_conn = DataConnector(preprocessed_data, 'trainable_embeddings_sent.pkl', trainable_embed)
	oov_conn.save_pickle()

	t1 = time.time()
	print("Initialize embedding matrix is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def create_embeddings_sent_fsoftmax(params):

	t0 = time.time()

	print("Initializing embedding matrix of in-domain and out-of-vocabulary...")
	sys.stdout.flush()

	glove_path = params['glove_path']
	glove_name = params['glove_name']
	glove_w2v = params['glove_w2v']
	preprocessed_data = params['preprocessed_data']
	embedding_dim = ['embedding_dim']

	id_word_conn = DataConnector(preprocessed_data, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	id_word_conn.read_pickle()
	id_word = id_word_conn.read_file

	from gensim.models.keyedvectors import KeyedVectors
	glove = KeyedVectors.load_word2vec_format(os.path.join(preprocessed_data, glove_w2v))

	print("Initializing nontrainable embeddings...")
	sys.stdout.flush()

	matrix_glove = PrepareEmbedding()
	nontrainable_embed = matrix_glove.create_nontrainable(id_word, glove, 100)

	print("Initializing trainable embeddings...")
	sys.stdout.flush()

	matrix_oov = PrepareEmbedding()
	trainable_embed = matrix_oov.create_trainable(id_word, 100)

	pretrained_conn = DataConnector(preprocessed_data, 'nontrainable_embeddings_sent_fsoftmax.pkl', nontrainable_embed)
	pretrained_conn.save_pickle()

	oov_conn = DataConnector(preprocessed_data, 'trainable_embeddings_sent_fsoftmax.pkl', trainable_embed)
	oov_conn.save_pickle()

	t1 = time.time()
	print("Initialize embedding matrix is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()