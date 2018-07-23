import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from datetime import datetime
import time
import random
from math import log
import json
from collections import OrderedDict


from utils.data_preprocessing_v2 import Preprocessing
from utils.reading_files import ReadingFiles
from utils.indexing import Indexing
from utils.data_connector import DataConnector
from utils.sequences_processing import SequenceProcessing
from utils.true_keyphrases import TrueKeyphrases


def reading_(params):

	train_path = params['train_path']
	valid_path = params['valid_path']
	test_path = params['test_path']

	# separated folders for training, validation, and test set

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Reading raw data...")
	sys.stdout.flush()

	## from training set
	read_data = ReadingFiles(train_path, 'inspec_train_doc_keyphrases.pkl')
	read_data.listing_files()
	read_data.reading_inspec()
	read_data.merging_inspec()
	# raw text data is stored in python dictionary format
	read_data.save_files()

	t1 = time.time()
	print("Reading raw training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

	t2 = time.time()
	print("Reading raw validation data...")
	sys.stdout.flush()

	## from validation set
	read_data = ReadingFiles(valid_path, 'inspec_val_doc_keyphrases.pkl')
	read_data.listing_files()
	read_data.reading_inspec()
	read_data.merging_inspec()
	# raw text data is stored in python dictionary format
	read_data.save_files()

	t3 = time.time()
	print("Reading raw validation data done in %.3fsec" % (t3 - t2))
	sys.stdout.flush()

	t4 = time.time()
	print("Reading raw test data...")
	sys.stdout.flush()

	## from validation set
	read_data = ReadingFiles(test_path, 'inspec_test_doc_keyphrases.pkl')
	read_data.listing_files()
	read_data.reading_inspec()
	read_data.merging_inspec()
	# raw text data is stored in python dictionary format
	read_data.save_files()

	t5 = time.time()
	print("Reading raw test data done in %.3fsec" % (t5 - t4))
	sys.stdout.flush()


def preprocessing_train(params):

	train_path = params['train_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw training data...")
	sys.stdout.flush()

	data_connector = DataConnector(train_path, 'inspec_train_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.preprocess_in(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenize_in(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)
	all_tokens = prep.get_all_tokens(input_tokens, output_tokens)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(train_path, 'train_input_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(train_path, 'train_output_tokens.npy', output_tokens)
	out_connector.save_numpys()
	tokens_connector = DataConnector(train_path, 'train_tokens.npy', all_tokens)
	tokens_connector.save_numpys()

	t1 = time.time()
	print("Preprocessing raw training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_sent_train(params):

	train_path = params['train_path']
	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw training data...")
	sys.stdout.flush()

	data_connector = DataConnector(train_path, 'inspec_train_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.split_sent(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenized_sent(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', output_tokens)
	out_connector.save_numpys()
	
	t1 = time.time()
	print("Preprocessing raw training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_valid(params):

	valid_path = params['valid_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw validation data...")
	sys.stdout.flush()

	data_connector = DataConnector(valid_path, 'inspec_val_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.preprocess_in(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenize_in(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)
	all_tokens = prep.get_all_tokens(input_tokens, output_tokens)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(valid_path, 'val_input_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(valid_path, 'val_output_tokens.npy', output_tokens)
	out_connector.save_numpys()
	tokens_connector = DataConnector(valid_path, 'val_tokens.npy', all_tokens)
	tokens_connector.save_numpys()

	t1 = time.time()
	print("Preprocessing raw validation data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_sent_valid(params):

	valid_path = params['valid_path']
	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw validation data...")
	sys.stdout.flush()

	data_connector = DataConnector(valid_path, 'inspec_val_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.split_sent(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenized_sent(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', output_tokens)
	out_connector.save_numpys()
	
	t1 = time.time()
	print("Preprocessing raw validation data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def preprocessing_test(params):

	test_path = params['test_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw test data...")
	sys.stdout.flush()

	data_connector = DataConnector(test_path, 'inspec_test_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.preprocess_in(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenize_in(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)
	all_tokens = prep.get_all_tokens(input_tokens, output_tokens)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(test_path, 'test_input_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(test_path, 'test_output_tokens.npy', output_tokens)
	out_connector.save_numpys()
	tokens_connector = DataConnector(test_path, 'test_tokens.npy', all_tokens)
	tokens_connector.save_numpys()

	t1 = time.time()
	print("Preprocessing raw test data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()
	
def preprocessing_sent_test(params):

	test_path = params['test_path']
	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing raw test data...")
	sys.stdout.flush()

	data_connector = DataConnector(test_path, 'inspec_test_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[2]
		
		in_text.append(text)
		out_keyphrases.append(kps)

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()

	print("\n in_text[0]: %s\n"%(in_text[0]))
	sys.stdout.flush()
	print("\n out_keyphrases[0]: %s\n"%(out_keyphrases[0]))
	sys.stdout.flush()

	prep = Preprocessing()
	prep_inputs = prep.split_sent(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenized_sent(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)

	# without splitting data into training and test set
	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', output_tokens)
	out_connector.save_numpys()
	

	t1 = time.time()
	print("Preprocessing raw test data done in %.3fsec" % (t1 - t0))
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

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in = train_in_connector.read_file

	val_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	val_in_connector.read_numpys()
	val_in = val_in_connector.read_file

	test_in_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_connector.read_numpys()
	test_in = test_in_connector.read_file

	sent_in = np.concatenate((train_in, val_in, test_in))

	
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

	print("average number of words per sentences: %s"%avg_len_sent)
	print("standard deviation number of words per sentences: %s"%std_len_sent)
	print("max number of words per sentences: %s"%max_len_sent)

	t1 = time.time()
	print("Computing stats done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def indexing_(params):

	data_path = params['data_path']
	train_path = params['train_path']
	valid_path = params['valid_path']
	test_path = params['test_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Vocabulary indexing...")
	sys.stdout.flush()

	'''
	read all tokens from training, validation, and testing set
	to create vocabulary index of word tokens
	'''

	train_tokens_connector = DataConnector(train_path, 'train_tokens.npy', data=None)
	train_tokens_connector.read_numpys()
	train_tokens = train_tokens_connector.read_file

	print("\n train_tokens[:10]: %s\n"%(train_tokens[:10]))
	sys.stdout.flush()

	valid_tokens_connector = DataConnector(valid_path, 'val_tokens.npy', data=None)
	valid_tokens_connector.read_numpys()
	valid_tokens = valid_tokens_connector.read_file

	print("\n valid_tokens[:10]: %s\n"%(valid_tokens[:10]))
	sys.stdout.flush()

	test_tokens_connector = DataConnector(test_path, 'test_tokens.npy', data=None)
	test_tokens_connector.read_numpys()
	test_tokens = test_tokens_connector.read_file

	print("\n test_tokens[:10]: %s\n"%(test_tokens[:10]))
	sys.stdout.flush()

	all_tokens = np.concatenate((train_tokens, valid_tokens, test_tokens))


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)
	
	term_freq_conn = DataConnector(data_path, 'term_freq.pkl', term_freq)
	term_freq_conn.save_pickle()
	indices_words_conn = DataConnector(data_path, 'indices_words.pkl', indices_words)
	indices_words_conn.save_pickle()
	words_indices_conn = DataConnector(data_path, 'words_indices.pkl', words_indices)
	words_indices_conn.save_pickle()

	print("\nvocabulary size: %s\n"%len(indices_words))
	sys.stdout.flush()

	print("\n indices_words[:10]: %s\n"%list(indices_words.items())[:10])
	sys.stdout.flush()

	t1 = time.time()
	print("Indexing done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()



def transform_train(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_kp20k, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_kp20k, 'all_wordidx_vocabulary.pkl', data=None)
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

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
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

	print("Transforming training set into integer sequences")
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
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
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
	preprocessed_v2 = params['preprocessed_v2']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary.pkl', data=None)
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

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(preprocessed_data, 'X_train.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_v1_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
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
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(preprocessed_data, 'X_train_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_fsoftmax.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_v2_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
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

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(preprocessed_data, 'X_train_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_fsoftmax.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(kp20k_path, 'all_indices_words_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_r3.pkl', data=None)
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

	print("\nshape of train_input_tokens in training set: %s\n"%str(np.array(train_in_tokens).shape))
	print("\nshape of train_output_tokens in training set: %s\n"%str(np.array(train_out_tokens).shape))


	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	X_train_pad = sequences_processing.pad_sequences_in(encoder_length, X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("\nshape of X_train in training set: %s\n"%str(np.array(X_train).shape))
	print("\nshape of X_train_pad in training set: %s\n"%str(np.array(X_train_pad).shape))

	print("\nshape of y_train_in in training set: %s\n"%str(np.array(y_train_in).shape))
	print("\nshape of y_train_out in training set: %s\n"%str(np.array(y_train_out).shape))


	x_in_connector = DataConnector(data_path, 'X_train_r3.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_train_pad_r3.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in_r3.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out_r3.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_train(params):

	data_path = params['data_path']
	train_path = params['train_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

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

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_train_sent.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_train_pad_sent.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_sent_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_sent_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_train_fsoftmax_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
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

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_sent_fsoftmax.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_sent_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_sent_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_train_fsoftmax_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent_fsoftmax.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_sent_fsoftmax.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_sent_in_fsoftmax.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_sent_out_fsoftmax.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_train_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
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

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_sent.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_sent_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_sent_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_train_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_train_sent.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_train_pad_sent.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_train_sent_in.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_train_sent_out.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_train_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(kp20k_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_input_sent_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_output_sent_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_train_sent_r3.npy', X_train)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_train_pad_sent_r3.npy', X_train_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_sent_in_r3.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_sent_out_r3.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()



def transform_valid(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_kp20k, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_kp20k, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

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


def transform_valid_v1_fsoftmax(params):

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

	print("Transforming validation set into integer sequences")
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

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

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


def transform_valid_v2_fsoftmax(params):

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

	print("Transforming validation set into integer sequences")
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

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

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

	print("Transforming validation set into integer sequences")
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

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

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


def transform_valid_v2(params):

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

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

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

def transform_valid_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(kp20k_path, 'all_indices_words_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.intexts_to_integers(valid_in_tokens)
	X_valid_pad = sequences_processing.pad_sequences_in(encoder_length, X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_valid_r3.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_valid_pad_r3.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_in_r3.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_out_r3.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_valid(params):

	data_path = params['data_path']
	valid_path = params['valid_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

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

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_valid_sent.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_valid_pad_sent.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_sent_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_sent_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_valid_fsoftmax_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_sent_in_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_sent_out_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_valid_fsoftmax_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
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

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent_fsoftmax.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent_fsoftmax.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_sent_in_fsoftmax.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_sent_out_fsoftmax.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_valid_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
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

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_sent_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_sent_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_valid_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(preprocessed_data, 'X_valid_sent.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_valid_pad_sent.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_valid_sent_in.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_valid_sent_out.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_valid_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming validation set into integer sequences")
	sys.stdout.flush()

	'''
	read stored vocabulary index
	'''

	vocab = DataConnector(kp20k_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	valid_in_connector = DataConnector(data_path, 'val_input_sent_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file
	valid_out_connector = DataConnector(data_path, 'val_output_sent_tokens.npy', data=None)
	valid_out_connector.read_numpys()
	valid_out_tokens = valid_out_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_valid = sequences_processing.in_sents_to_integers(valid_in_tokens, max_sents)
	X_valid_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_valid)
	y_valid_in, y_valid_out = sequences_processing.outtexts_to_integers(valid_out_tokens)

	x_in_connector = DataConnector(data_path, 'X_valid_sent_r3.npy', X_valid)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_valid_pad_sent_r3.npy', X_valid_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_valid_sent_in_r3.npy', y_valid_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_valid_sent_out_r3.npy', y_valid_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming validation set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_test(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_kp20k, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_kp20k, 'all_wordidx_vocabulary.pkl', data=None)
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

	'''
	transforming texts into integer sequences
	'''
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


def transform_test_v1_fsoftmax(params):

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

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_fsoftmax.pkl', data=None)
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

	'''
	transforming texts into integer sequences
	'''
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

def transform_test_v2_fsoftmax(params):

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

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

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

	'''
	transforming texts into integer sequences
	'''
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

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

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

	'''
	transforming texts into integer sequences
	'''
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


def transform_test_v2(params):

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

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary.pkl', data=None)
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

	'''
	transforming texts into integer sequences
	'''
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

def transform_test_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(kp20k_path, 'all_indices_words_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_r3.pkl', data=None)
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

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	X_test_pad = sequences_processing.pad_sequences_in(encoder_length, X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(data_path, 'X_test_r3.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_test_pad_r3.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_in_r3.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_out_r3.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_test(params):

	data_path = params['data_path']
	test_path = params['test_path']
	max_sents = params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(data_path, 'X_test_sent.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_test_pad_sent.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_sent_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_sent_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_test_fsoftmax_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_sent_in_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_sent_out_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_test_fsoftmax_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent_fsoftmax.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent_fsoftmax.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_sent_in_fsoftmax.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_sent_out_fsoftmax.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_test_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_sent_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_sent_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_test_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(preprocessed_data, 'X_test_sent.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(preprocessed_data, 'X_test_pad_sent.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_test_sent_in.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_test_sent_out.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_test_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents = params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(kp20k_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	test_in_tokens_connector = DataConnector(data_path, 'test_input_sent_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'test_output_sent_tokens.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	'''
	transforming texts into integer sequences
	'''
	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.in_sents_to_integers(test_in_tokens, max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents= max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)


	x_in_connector = DataConnector(data_path, 'X_test_sent_r3.npy', X_test)
	x_in_connector.save_numpys()
	x_pad_in_connector = DataConnector(data_path, 'X_test_pad_sent_r3.npy', X_test_pad)
	x_pad_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_sent_in_r3.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_sent_out_r3.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_train(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing training data to train the model...")
	sys.stdout.flush()

	x_train_connector = DataConnector(preprocessed_data, 'X_train.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(preprocessed_data, 'y_train_in.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(preprocessed_data, 'y_train_out.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=encoder_length, decoder_length=decoder_length)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))

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


def pair_train_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing training data to train the model...")
	sys.stdout.flush()

	x_train_connector = DataConnector(data_path, 'X_train_r2.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_in_r2.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_out_r2.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=encoder_length, decoder_length=decoder_length)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_train_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train_r2.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in_r2.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out_r2.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_sent_train(params):

	data_path = params['data_path']
	train_path = params['train_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing training data to train the model...")
	sys.stdout.flush()

	x_train_connector = DataConnector(data_path, 'X_train_sent.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_sent_in.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_sent_out.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_train_, y_pair_train_in_, y_pair_train_out_ = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	x_pair_train = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_train_)
	y_pair_train_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_train_in_)
	y_pair_train_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_train_out_)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_train_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train_sent.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_sent_in.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_sent_out.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_sent_train_sub(params):

	data_path = params['data_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing training data to train the model...")
	sys.stdout.flush()

	x_train_connector = DataConnector(data_path, 'X_train_sent_r2.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_sent_in_r2.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_sent_out_r2.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_train_, y_pair_train_in_, y_pair_train_out_ = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	x_pair_train = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_train_)
	y_pair_train_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_train_in_)
	y_pair_train_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_train_out_)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_train_sent_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train_sent_r2.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_sent_in_r2.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_sent_out_r2.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_valid(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
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

	X_valid_connector = DataConnector(preprocessed_data, 'X_valid.npy', data=None)
	X_valid_connector.read_numpys()
	X_valid = X_valid_connector.read_file

	y_valid_in_connector = DataConnector(preprocessed_data, 'y_valid_in.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(preprocessed_data, 'y_valid_out.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)
	x_pair_val, y_pair_val_in, y_pair_val_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	
	print("\nshape of x_pair in val set: %s\n"%str(x_pair_val.shape))
	print("\nshape of y_pair_in in val set: %s\n"%str(y_pair_val_in.shape))
	print("\nshape of y_pair_out in val set: %s\n"%str(y_pair_val_out.shape))

	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_test.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_val.npy', x_pair_val)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_val_in.npy', y_pair_val_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_val_out.npy', y_pair_val_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_valid_sub(params):

	data_path = params['data_path']
	
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing validation set...")
	sys.stdout.flush()

	X_valid_connector = DataConnector(data_path, 'X_valid_r2.npy', data=None)
	X_valid_connector.read_numpys()
	X_valid = X_valid_connector.read_file

	y_valid_in_connector = DataConnector(data_path, 'y_valid_in_r2.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(data_path, 'y_valid_out_r2.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)
	x_pair_val, y_pair_val_in, y_pair_val_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	
	print("\nshape of x_pair in val set: %s\n"%str(x_pair_val.shape))
	print("\nshape of y_pair_in in val set: %s\n"%str(y_pair_val_in.shape))
	print("\nshape of y_pair_out in val set: %s\n"%str(y_pair_val_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_val_r2.npy', x_pair_val)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_val_in_r2.npy', y_pair_val_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_val_out_r2.npy', y_pair_val_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_sent_valid(params):

	data_path = params['data_path']
	valid_path = params['valid_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing validation set...")
	sys.stdout.flush()

	X_valid_connector = DataConnector(data_path, 'X_valid_sent.npy', data=None)
	X_valid_connector.read_numpys()
	X_valid = X_valid_connector.read_file

	y_valid_in_connector = DataConnector(data_path, 'y_valid_sent_in.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(data_path, 'y_valid_sent_out.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	
	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_val_, y_pair_val_in_, y_pair_val_out_ = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	x_pair_val = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_val_)
	y_pair_val_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_val_in_)
	y_pair_val_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_val_out_)

	print("\nshape of x_pair in val set: %s\n"%str(x_pair_val.shape))
	print("\nshape of y_pair_in in val set: %s\n"%str(y_pair_val_in.shape))
	print("\nshape of y_pair_out in val set: %s\n"%str(y_pair_val_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_val_sent.npy', x_pair_val)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_val_sent_in.npy', y_pair_val_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_val_sent_out.npy', y_pair_val_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_sent_valid_sub(params):

	data_path = params['data_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing validation set...")
	sys.stdout.flush()

	X_valid_connector = DataConnector(data_path, 'X_valid_sent_r2.npy', data=None)
	X_valid_connector.read_numpys()
	X_valid = X_valid_connector.read_file

	y_valid_in_connector = DataConnector(data_path, 'y_valid_sent_in_r2.npy', data=None)
	y_valid_in_connector.read_numpys()
	y_valid_in = y_valid_in_connector.read_file

	y_valid_out_connector = DataConnector(data_path, 'y_valid_sent_out_r2.npy', data=None)
	y_valid_out_connector.read_numpys()
	y_valid_out = y_valid_out_connector.read_file

	
	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_val_, y_pair_val_in_, y_pair_val_out_ = sequences_processing.pairing_data_(X_valid, y_valid_in, y_valid_out)

	x_pair_val = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_val_)
	y_pair_val_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_val_in_)
	y_pair_val_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_val_out_)

	print("\nshape of x_pair in val set: %s\n"%str(x_pair_val.shape))
	print("\nshape of y_pair_in in val set: %s\n"%str(y_pair_val_in.shape))
	print("\nshape of y_pair_out in val set: %s\n"%str(y_pair_val_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_sent_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_val_sent_r2.npy', x_pair_val)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_val_sent_in_r2.npy', y_pair_val_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_val_sent_out_r2.npy', y_pair_val_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing validation set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_test(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	X_test_connector = DataConnector(preprocessed_data, 'X_test.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file



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

def pair_test_sub(params):

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	X_test_connector = DataConnector(data_path, 'X_test_r2.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_in_r2.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_out_r2.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file



	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_r2_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test_r2.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_in_r2.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_out_r2.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_sent_test(params):

	data_path = params['data_path']
	test_path = params['test_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	X_test_connector = DataConnector(data_path, 'X_test_sent.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_sent_in.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_sent_out.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file


	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_test_, y_pair_test_in_, y_pair_test_out_ = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_test_)
	y_pair_test_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_test_in_)
	y_pair_test_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_test_out_)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_sent.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test_sent.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_sent_in.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_sent_out.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_sent_test_sub(params):

	data_path = params['data_path']
	max_sents = params['max_sents']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	X_test_connector = DataConnector(data_path, 'X_test_sent_r2.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_sent_in_r2.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_sent_out_r2.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file


	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_test_, y_pair_test_in_, y_pair_test_out_ = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=x_pair_test_)
	y_pair_test_in = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_test_in_)
	y_pair_test_out = sequences_processing.pad_sequences_in(max_len=decoder_length+1, sequences=y_pair_test_out_)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_sent_r2.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test_sent_r2.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_sent_in_r2.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_sent_out_r2.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing test set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

'''
Get average number of key phrases per document in corpus
'''

def compute_average_keyphrases(params):

	data_path = params['data_path']
	train_path = params['train_path']
	valid_path = params['valid_path']
	test_path = params['test_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing average key phrases per document...")
	sys.stdout.flush()

	# from training set
	train_kp_connector = DataConnector(train_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	# from validation set
	valid_kp_connector = DataConnector(valid_path, 'val_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	# from test set

	test_kp_connector = DataConnector(test_path, 'test_output_tokens.npy', data=None)
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
	train_path = params['train_path']
	valid_path = params['valid_path']
	test_path = params['test_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing statistics of key phrases per document...")
	sys.stdout.flush()

	# from training set
	train_kp_connector = DataConnector(train_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	# from validation set
	valid_kp_connector = DataConnector(valid_path, 'val_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	# from test set

	test_kp_connector = DataConnector(test_path, 'test_output_tokens.npy', data=None)
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
	train_path = params['train_path']
	valid_path = params['valid_path']
	test_path = params['test_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Computing presence or absence of key phrases per document...")
	sys.stdout.flush()

	# from training set

	train_in_connector = DataConnector(train_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file

	train_kp_connector = DataConnector(train_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	compute_presence_train = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	all_npresence_train, all_nabsence_train = compute_presence_train.compute_presence(train_in_tokens, train_kps)
	total_train = np.sum(all_npresence_train) + np.sum(all_nabsence_train)


	# from validation set

	valid_in_connector = DataConnector(valid_path, 'val_input_tokens.npy', data=None)
	valid_in_connector.read_numpys()
	valid_in_tokens = valid_in_connector.read_file

	valid_kp_connector = DataConnector(valid_path, 'val_output_tokens.npy', data=None)
	valid_kp_connector.read_numpys()
	valid_kps = valid_kp_connector.read_file

	compute_presence_val = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	all_npresence_val, all_nabsence_val = compute_presence_val.compute_presence(valid_in_tokens, valid_kps)
	total_val = np.sum(all_npresence_val) + np.sum(all_nabsence_val)

	# from test set

	test_in_tokens_connector = DataConnector(test_path, 'test_input_tokens.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file

	test_kp_connector = DataConnector(test_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	compute_presence_test = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
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




