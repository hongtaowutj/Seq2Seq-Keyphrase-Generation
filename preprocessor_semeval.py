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

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Reading raw training data...")
	sys.stdout.flush()

	data_path = params['data_path']
	train_path = params['train_path']
	test_path = params['test_path']

	read_data1 = ReadingFiles(train_path, 'semeval_train_doc_keyphrases.pkl')
	read_data1.listing_files()
	read_data1.reading_semeval()
	read_data1.merging_data()
	# raw text data is stored in python dictionary format
	read_data1.save_files()

	t1 = time.time()
	print("Reading raw training data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

	t2 = time.time()
	print("Reading raw test data...")
	sys.stdout.flush()

	read_data2 = ReadingFiles(test_path, 'semeval_test_doc_keyphrases.pkl')
	read_data2.listing_files()
	read_data2.reading_semeval()
	read_data2.merging_data()
	# raw text data is stored in python dictionary format
	read_data2.save_files()

	t3 = time.time()
	print("Reading raw test data done in %.3fsec" % (t3 - t2))
	sys.stdout.flush()


def preprocessing_semeval(params):

	data_path = params['data_path']
	train_path = params['train_path']
	test_path = params['test_path']

	semeval_train_connector = DataConnector(train_path, 'semeval_train_doc_keyphrases.pkl', data=None)
	semeval_train_connector.read_pickle()
	semeval_train_doc_topics = semeval_train_connector.read_file

	semeval_test_connector = DataConnector(test_path, 'semeval_test_doc_keyphrases.pkl', data=None)
	semeval_test_connector.read_pickle()
	semeval_test_doc_topics = semeval_test_connector.read_file

	train_in_text = []
	train_out_keyphrases = []

	for k,v in semeval_train_doc_topics.items():
		title = v[0]

		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		train_in_text.append(text)
		train_out_keyphrases.append(kps)

		

	test_in_text = []
	test_out_keyphrases = []
	for k,v in semeval_test_doc_topics.items():
		title = v[0]

		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		test_in_text.append(text)
		test_out_keyphrases.append(kps)


	train_prep = Preprocessing()
	train_prep_inputs = train_prep.preprocess_in(train_in_text)
	train_prep_outputs = train_prep.preprocess_out(train_out_keyphrases)
	train_input_tokens = train_prep.tokenize_in(train_prep_inputs)
	train_output_tokens = train_prep.tokenize_out(train_prep_outputs)
	train_tokens = train_prep.get_all_tokens(train_input_tokens, train_output_tokens)

	test_prep = Preprocessing()
	test_prep_inputs = test_prep.preprocess_in(test_in_text)
	test_prep_outputs = test_prep.preprocess_out(test_out_keyphrases)
	test_input_tokens = test_prep.tokenize_in(test_prep_inputs)
	test_output_tokens = test_prep.tokenize_out(test_prep_outputs)
	test_tokens = test_prep.get_all_tokens(test_input_tokens, test_output_tokens)

	all_tokens = np.concatenate((train_tokens, test_tokens))

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', train_input_tokens)
	train_in_connector.save_numpys()
	train_out_connector = DataConnector(data_path, 'train_output_tokens.npy', train_output_tokens)
	train_out_connector.save_numpys()

	test_in_connector = DataConnector(data_path, 'test_input_tokens.npy', test_input_tokens)
	test_in_connector.save_numpys()
	test_out_connector = DataConnector(data_path, 'test_output_tokens.npy', test_output_tokens)
	test_out_connector.save_numpys()

	tokens_connector = DataConnector(data_path, 'all_tokens.npy', all_tokens)
	tokens_connector.save_numpys()

def preprocessing_sent_semeval(params):

	data_path = params['data_path']
	train_path = params['train_path']
	test_path = params['test_path']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing into tokenized sentences...")
	sys.stdout.flush()

	semeval_train_connector = DataConnector(train_path, 'semeval_train_doc_keyphrases.pkl', data=None)
	semeval_train_connector.read_pickle()
	semeval_train_doc_topics = semeval_train_connector.read_file

	semeval_test_connector = DataConnector(test_path, 'semeval_test_doc_keyphrases.pkl', data=None)
	semeval_test_connector.read_pickle()
	semeval_test_doc_topics = semeval_test_connector.read_file

	train_in_text = []
	train_out_keyphrases = []

	for k,v in semeval_train_doc_topics.items():
		title = v[0]

		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		train_in_text.append(text)
		train_out_keyphrases.append(kps)

	print("train_in_text[0]: %s"%(train_in_text[0]))
	sys.stdout.flush()

	test_in_text = []
	test_out_keyphrases = []

	for k,v in semeval_test_doc_topics.items():
		title = v[0]

		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		test_in_text.append(text)
		test_out_keyphrases.append(kps)

	print("test_in_text[0]: %s"%(test_in_text[0]))
	sys.stdout.flush()

	train_prep = Preprocessing()
	train_prep_inputs = train_prep.split_sent(train_in_text)
	train_prep_outputs = train_prep.preprocess_out(train_out_keyphrases)
	train_input_tokens = train_prep.tokenized_sent(train_prep_inputs)
	train_output_tokens = train_prep.tokenize_out(train_prep_outputs)

	test_prep = Preprocessing()
	test_prep_inputs = test_prep.split_sent(test_in_text)
	test_prep_outputs = test_prep.preprocess_out(test_out_keyphrases)
	test_input_tokens = test_prep.tokenized_sent(test_prep_inputs)
	test_output_tokens = test_prep.tokenize_out(test_prep_outputs)

	sent_in = np.concatenate((train_input_tokens, test_input_tokens))

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

	train_in_connector = DataConnector(data_path, 'train_sent_input_tokens.npy', train_input_tokens)
	train_in_connector.save_numpys()
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', train_output_tokens)
	train_out_connector.save_numpys()

	test_in_connector = DataConnector(data_path, 'test_sent_input_tokens.npy', test_input_tokens)
	test_in_connector.save_numpys()
	test_out_connector = DataConnector(data_path, 'test_sent_output_tokens.npy', test_output_tokens)
	test_out_connector.save_numpys()
	
	t1 = time.time()
	print("Preprocessing sentences is done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def indexing_(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Vocabulary indexing...")
	sys.stdout.flush()


	tokens_connector = DataConnector(data_path, 'all_tokens.npy', data=None)
	tokens_connector.read_numpys()
	all_tokens = tokens_connector.read_file


	indexing = Indexing()
	term_freq, indices_words, words_indices = indexing.vocabulary_indexing(all_tokens)

	term_freq_conn = DataConnector(data_path, 'term_freq.pkl', term_freq)
	term_freq_conn.save_pickle()
	indices_words_conn = DataConnector(data_path, 'indices_words.pkl', indices_words)
	indices_words_conn.save_pickle()
	words_indices_conn = DataConnector(data_path, 'words_indices.pkl', words_indices)
	words_indices_conn.save_pickle()

	print("vocabulary size: %s"%len(indices_words))
	sys.stdout.flush()

	t1 = time.time()
	print("Indexing done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train_fsoftmax(params):

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

def transform_train(params):

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

def transform_train_self(params):

	data_path = params['data_path']

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

	x_in_connector = DataConnector(data_path, 'X_train_r3.npy', X_train)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(data_path, 'X_train_pad_r3.npy', X_train_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in_r3.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out_r3.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_train_sent_fsoftmax_v1(params):

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

	train_in_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()


	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

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
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

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

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()


	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

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

def transform_train_sent_v1(params):

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

	train_in_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()


	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

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
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

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

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()


	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

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

def transform_train_sent(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']

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

	vocab = DataConnector(kp20k_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	'''
	read tokenized data set
	'''

	train_in_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'train_sent_output_tokens.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(train_in_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(train_out_tokens)))
	sys.stdout.flush()


	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.in_sents_to_integers(in_texts=train_in_tokens, max_sents=max_sents)
	X_train_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_train)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(out_texts=train_out_tokens)

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

def transform_test_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
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

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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


def transform_test(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
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

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
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

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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


def transform_test_self(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
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

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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


def transform_test_sent_fsoftmax_v1(params):

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
	X_test = sequences_processing.in_sents_to_integers(in_texts=test_in_tokens, max_sents=max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(out_texts=test_out_tokens)

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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
	X_test = sequences_processing.in_sents_to_integers(in_texts=test_in_tokens, max_sents=max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(out_texts=test_out_tokens)

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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


def transform_test_sent_v1(params):

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
	X_test = sequences_processing.in_sents_to_integers(in_texts=test_in_tokens, max_sents=max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(out_texts=test_out_tokens)

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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
	X_test = sequences_processing.in_sents_to_integers(in_texts=test_in_tokens, max_sents=max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(out_texts=test_out_tokens)

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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

def transform_test_sent(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming test set into integer sequences")
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
	X_test = sequences_processing.in_sents_to_integers(in_texts=test_in_tokens, max_sents=max_sents)
	X_test_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents, sequences=X_test)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(out_texts=test_out_tokens)

	print("X_test_pad: %s"%str(X_test_pad.shape))
	print("\nnumber of X examples in test set: %s\n"%(len(X_test)))
	sys.stdout.flush()
	print("\nnumber of Y examples in test set: %s\n"%(len(y_test_in)))
	sys.stdout.flush()

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

def pair_train_(params):

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

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

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=encoder_length, decoder_length=decoder_length)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)


	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(data_path, 'doc_pair_train.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_train_self(params):

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Pairing training set...")
	sys.stdout.flush()

	'''
	read training set

	'''
	x_train_connector = DataConnector(data_path, 'X_train_r3.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_in_r3.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_out_r3.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	print("\n n-X_train: %s\n"%len(X_train))
	sys.stdout.flush()
	print("\n n-y_train_in: %s\n"%len(y_train_in))
	sys.stdout.flush()
	print("\n n-y_train_out: %s\n"%len(y_train_out))
	sys.stdout.flush()

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_train, y_train_in, y_train_out)

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(data_path, 'doc_pair_train_r3.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train_r3.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in_r3.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out_r3.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_test_(params):

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	'''
	read training set

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

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=encoder_length, decoder_length=decoder_length)
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

def pair_test_self(params):

	data_path = params['data_path']
	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	'''
	read training set

	'''
	X_test_connector = DataConnector(data_path, 'X_test_r3.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_in_r3.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_out_r3.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X_test, y_test_in, y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)

	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_r3.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test_r3.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_in_r3.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_out_r3.npy', y_pair_test_out)
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

	# from test set

	test_kp_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	all_keyphrases = np.concatenate((train_kps, test_kps))


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

	# from test set

	test_kp_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	test_kp_connector.read_numpys()
	test_kps = test_kp_connector.read_file

	all_keyphrases = np.concatenate((train_kps, test_kps))

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



	print("Maximum number of key phrases per document in corpus: %s" %max_kps)
	sys.stdout.flush()
	print("Average number of key phrases per document in corpus: %s" %mean_kps)
	sys.stdout.flush()
	print("Standard Deviation of number of key phrases per document in corpus: %s" %std_kps)
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

	train_in_connector = DataConnector(data_path, 'train_input_tokens.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file

	train_kp_connector = DataConnector(data_path, 'train_output_tokens.npy', data=None)
	train_kp_connector.read_numpys()
	train_kps = train_kp_connector.read_file

	compute_presence_train = SequenceProcessing()
	all_npresence_train, all_nabsence_train = compute_presence_train.compute_presence(train_in_tokens, train_kps)
	total_train = np.sum(all_npresence_train) + np.sum(all_nabsence_train)


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

	n_presence = np.sum(all_npresence_train) + np.sum(all_npresence_test)
	n_absence = np.sum(all_nabsence_train) + np.sum(all_nabsence_test)
	total = total_train +  total_test

	persen_absence = n_absence / total
	persen_presence = n_presence / total

	print(" Absent key phrase: %s" %persen_absence)
	sys.stdout.flush()
	print(" Present key phrase: %s" %persen_presence)
	sys.stdout.flush()


	t1 = time.time()
	print("Computing presence or absence of key phrases per document done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()