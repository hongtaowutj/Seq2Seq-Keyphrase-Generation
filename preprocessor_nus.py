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

	data_path = params['data_path']

	t0 = time.time()
	print("Reading raw data...")
	sys.stdout.flush()

	read_data = ReadingFiles(data_path, 'nus_doc_keyphrases.pkl')
	read_data.listing_files()
	read_data.construct_dirinfos()
	read_data.merging_keys()
	read_data.reading_texts()
	read_data.merging_data()
	# raw text data is stored in python dictionary format
	read_data.save_files()

	t1 = time.time()
	print("Reading raw data done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def preprocessing_(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing data...")
	sys.stdout.flush()

	data_connector = DataConnector(data_path, 'nus_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	nus_data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in nus_data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		in_text.append(text)
		out_keyphrases.append(kps)

	in_connector = DataConnector(data_path, 'raw_in.npy', in_text)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'raw_out.npy', out_keyphrases)
	out_connector.save_numpys()

	print("\nnumber of examples in raw data inputs: %s\n"%(len(in_text)))
	sys.stdout.flush()
	print("\nnumber of examples in raw data outputs: %s\n"%(len(out_keyphrases)))
	sys.stdout.flush()


	prep = Preprocessing()
	prep_inputs = prep.preprocess_in(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenize_in(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)
	all_tokens = prep.get_all_tokens(input_tokens, output_tokens)

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	print("\n input_tokens[0]: %s\n"%(input_tokens[0]))
	sys.stdout.flush()
	print("\n output_tokens[0]: %s\n"%(output_tokens[0]))
	sys.stdout.flush()

	in_connector = DataConnector(data_path, 'input_tokens.npy', input_tokens)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'output_tokens.npy', output_tokens)
	out_connector.save_numpys()
	tokens_connector = DataConnector(data_path, 'all_tokens.npy', all_tokens)
	tokens_connector.save_numpys()

	# splitting into training and test set
	n_train = int(0.8 * len(input_tokens))
	in_train = input_tokens[:n_train]
	out_train = output_tokens[:n_train]
	in_test = input_tokens[n_train:len(input_tokens)]
	out_test = output_tokens[n_train:len(input_tokens)]

	print("\nnumber of examples in training set: %s\n"%(len(in_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(in_test)))
	sys.stdout.flush()

	in_train_connector = DataConnector(data_path, 'in_train.npy', in_train)
	in_train_connector.save_numpys()
	out_train_connector = DataConnector(data_path, 'out_train.npy', out_train)
	out_train_connector.save_numpys()
	in_test_connector = DataConnector(data_path, 'in_test.npy', in_test)
	in_test_connector.save_numpys()
	out_test_connector = DataConnector(data_path, 'out_test.npy', out_test)
	out_test_connector.save_numpys()

	t1 = time.time()
	print("Preprocessing done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def preprocessing_sent_(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()
	print("Preprocessing data...")
	sys.stdout.flush()

	data_connector = DataConnector(data_path, 'nus_doc_keyphrases.pkl', data=None)
	data_connector.read_pickle()
	nus_data = data_connector.read_file

	in_text = []
	out_keyphrases = []

	for k,v in nus_data.items():
		title = v[0]
		abstract = v[1]
		text = title + " . " + abstract
		kps = v[3]
		in_text.append(text)
		out_keyphrases.append(kps)

	in_connector = DataConnector(data_path, 'raw_in.npy', in_text)
	in_connector.save_numpys()
	out_connector = DataConnector(data_path, 'raw_out.npy', out_keyphrases)
	out_connector.save_numpys()

	prep = Preprocessing()
	prep_inputs = prep.split_sent(in_text)
	prep_outputs = prep.preprocess_out(out_keyphrases)
	input_tokens = prep.tokenized_sent(prep_inputs)
	output_tokens = prep.tokenize_out(prep_outputs)

	print("\nnumber of examples in preprocessed data inputs: %s\n"%(len(input_tokens)))
	sys.stdout.flush()

	print("\nnumber of examples in preprocessed data outputs: %s\n"%(len(output_tokens)))
	sys.stdout.flush()

	prep_in_connector = DataConnector(data_path, 'sent-input_tokens.npy', input_tokens)
	prep_in_connector.save_numpys()
	prep_out_connector = DataConnector(data_path, 'sent-output_tokens.npy', output_tokens)
	prep_out_connector.save_numpys()

	# splitting into training and test set
	n_train = int(0.8 * len(input_tokens))
	in_train = input_tokens[:n_train]
	out_train = output_tokens[:n_train]
	in_test = input_tokens[n_train:len(input_tokens)]
	out_test = output_tokens[n_train:len(input_tokens)]

	print("\nnumber of examples in training set: %s\n"%(len(in_train)))
	sys.stdout.flush()
	print("\nnumber of examples in test set: %s\n"%(len(in_test)))
	sys.stdout.flush()

	in_train_connector = DataConnector(data_path, 'sent_in_train.npy', in_train)
	in_train_connector.save_numpys()
	out_train_connector = DataConnector(data_path, 'sent_out_train.npy', out_train)
	out_train_connector.save_numpys()
	in_test_connector = DataConnector(data_path, 'sent_in_test.npy', in_test)
	in_test_connector.save_numpys()
	out_test_connector = DataConnector(data_path, 'sent_out_test.npy', out_test)
	out_test_connector.save_numpys()
	
	t1 = time.time()
	print("Preprocessing done in %.3fsec" % (t1 - t0))
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

	sent_in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	sent_in_connector.read_numpys()
	sent_in = sent_in_connector.read_file

	
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
	print("Computing stats done in %.3fsec" % (t1 - t0))
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

	print("\nvocabulary size: %s\n"%len(indices_words))
	sys.stdout.flush()

	t1 = time.time()
	print("Indexing done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_all(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_kp20k, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_kp20k, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus.npy', x_in)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_v1_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	x_in_pad = sequences_processing.pad_sequences_in(encoder_length, x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_fsoftmax.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_fsoftmax.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_fsoftmax.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_fsoftmax.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_v2_fsoftmax(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	x_in_pad = sequences_processing.pad_sequences_in(encoder_length, x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_fsoftmax.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_fsoftmax.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_fsoftmax.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_fsoftmax.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_v1(params):

	data_path = params['data_path']
	preprocessed_data = params['preprocessed_data']
	preprocessed_v2 = params['preprocessed_v2']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']


	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	x_in_pad = sequences_processing.pad_sequences_in(encoder_length, x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_all_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	preprocessed_kp20k = params['preprocessed_kp20k'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_kp20k, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_kp20k, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	x_in_pad = sequences_processing.pad_sequences_in(encoder_length, x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_v2(params):

	data_path = params['data_path']
	preprocessed_v2 = params['preprocessed_v2'] 
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.intexts_to_integers(input_tokens)
	x_in_pad = sequences_processing.pad_sequences_in(encoder_length, x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_v1(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'sent-output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.in_sents_to_integers(in_texts=input_tokens, max_sents=max_sents)
	x_in_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(out_texts=output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_sent.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_sent.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_sent.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_sent.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_v2(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'sent-output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.in_sents_to_integers(in_texts=input_tokens, max_sents=max_sents)
	x_in_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(out_texts=output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_sent.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_sent.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_sent.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_sent.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_v1_fsoftmax(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_indices_words_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_words_indices_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'sent-output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.in_sents_to_integers(in_texts=input_tokens, max_sents=max_sents)
	x_in_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(out_texts=output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_sent_fsoftmax.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_sent_fsoftmax.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_sent_fsoftmax.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_sent_fsoftmax.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def transform_sent_v2_fsoftmax(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']
	preprocessed_v2 = params['preprocessed_v2']
	preprocessed_data = params['preprocessed_data']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(preprocessed_v2, 'all_idxword_vocabulary_sent_fsoftmax.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(preprocessed_v2, 'all_wordidx_vocabulary_sent_fsoftmax.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'sent-output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.in_sents_to_integers(in_texts=input_tokens, max_sents=max_sents)
	x_in_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(out_texts=output_tokens)	

	x_in_connector = DataConnector(preprocessed_data, 'X_nus_sent_fsoftmax.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(preprocessed_data, 'X_nus_pad_sent_fsoftmax.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus_sent_fsoftmax.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus_sent_fsoftmax.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_sent_sub(params):

	data_path = params['data_path']
	kp20k_path = params['kp20k_path']
	max_sents= params['max_sents']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming all data set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(kp20k_path, 'all_indices_words_sent_r3.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(kp20k_path, 'all_words_indices_sent_r3.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file


	in_connector = DataConnector(data_path, 'sent-input_tokens.npy', data=None)
	in_connector.read_numpys()
	input_tokens = in_connector.read_file
	out_connector = DataConnector(data_path, 'sent-output_tokens.npy', data=None)
	out_connector.read_numpys()
	output_tokens = out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	x_in = sequences_processing.in_sents_to_integers(in_texts=input_tokens, max_sents=max_sents)
	x_in_pad = sequences_processing.pad_sequences_sent_in(max_len=encoder_length, max_sents=max_sents,sequences=x_in)
	y_in, y_out = sequences_processing.outtexts_to_integers(out_texts=output_tokens)	

	x_in_connector = DataConnector(data_path, 'X_nus_sent_r3.npy', x_in)
	x_in_connector.save_numpys()
	x_in_pad_connector = DataConnector(data_path, 'X_nus_pad_sent_r3.npy', x_in_pad)
	x_in_pad_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_in_nus_sent_r3.npy', y_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_out_nus_sent_r3.npy', y_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming data set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def transform_train(params):

	data_path = params['data_path']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']

	print("\n=========\n")

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Transforming training set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	train_in_connector = DataConnector(data_path, 'in_train.npy', data=None)
	train_in_connector.read_numpys()
	train_in_tokens = train_in_connector.read_file
	train_out_connector = DataConnector(data_path, 'out_train.npy', data=None)
	train_out_connector.read_numpys()
	train_out_tokens = train_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_train = sequences_processing.intexts_to_integers(train_in_tokens)
	y_train_in, y_train_out = sequences_processing.outtexts_to_integers(train_out_tokens)

	print("X train shape: %s"%str(np.array(X_train).shape)) 
	sys.stdout.flush()

	print("y_train_in shape: %s"%str(np.array(y_train_in).shape)) 
	sys.stdout.flush()

	print("y_train_out shape: %s"%str(np.array(y_train_out).shape)) 
	sys.stdout.flush()

	x_in_connector = DataConnector(data_path, 'X_train_nus.npy', X_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_train_in_nus.npy', y_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_train_out_nus.npy', y_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming training set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
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

	print("Transforming test set into integer sequences")
	sys.stdout.flush()

	vocab = DataConnector(data_path, 'indices_words.pkl', data=None)
	vocab.read_pickle()
	indices_words = vocab.read_file

	reversed_vocab = DataConnector(data_path, 'words_indices.pkl', data=None)
	reversed_vocab.read_pickle()
	words_indices = reversed_vocab.read_file

	test_in_tokens_connector = DataConnector(data_path, 'in_test.npy', data=None)
	test_in_tokens_connector.read_numpys()
	test_in_tokens = test_in_tokens_connector.read_file
	test_out_tokens_connector = DataConnector(data_path, 'out_test.npy', data=None)
	test_out_tokens_connector.read_numpys()
	test_out_tokens = test_out_tokens_connector.read_file

	sequences_processing = SequenceProcessing(indices_words, words_indices, encoder_length, decoder_length)
	X_test = sequences_processing.intexts_to_integers(test_in_tokens)
	y_test_in, y_test_out = sequences_processing.outtexts_to_integers(test_out_tokens)	

	print("X_test shape: %s"%str(np.array(X_test).shape)) 
	sys.stdout.flush()

	print("y_test_in shape: %s"%str(np.array(y_test_in).shape)) 
	sys.stdout.flush()

	print("y_test_out shape: %s"%str(np.array(y_test_out).shape)) 
	sys.stdout.flush()

	x_in_connector = DataConnector(data_path, 'X_test_nus.npy', X_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_test_in_nus.npy', y_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_test_out_nus.npy', y_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Transforming test set into integer sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_all(params):

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

	print("Pairing data to train the model...")
	sys.stdout.flush()

	x_connector = DataConnector(preprocessed_data, 'X_nus.npy', data=None)
	x_connector.read_numpys()
	X = x_connector.read_file

	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', data=None)
	y_in_connector.read_numpys()
	y_in_ = y_in_connector.read_file

	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', data=None)
	y_out_connector.read_numpys()
	y_out_ = y_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=None, decoder_length=None)
	doc_pair, x_pair_, y_pair_in_, y_pair_out_ = sequences_processing.pairing_data_(X, y_in_, y_out_)

	x_pair, y_pair_in, y_pair_out = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair_, y_pair_in_, y_pair_out_)


	print("\nshape of x_pair: %s\n"%str(x_pair.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in: %s\n"%str(y_pair_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out: %s\n"%str(y_pair_out.shape))
	sys.stdout.flush()
	
	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_nus.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_nus.npy', x_pair)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_in_nus.npy', y_pair_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_out_nus.npy', y_pair_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()

def pair_all_sub(params):

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

	print("Pairing data to train the model...")
	sys.stdout.flush()

	x_connector = DataConnector(preprocessed_data, 'X_nus.npy', data=None)
	x_connector.read_numpys()
	X = x_connector.read_file

	y_in_connector = DataConnector(preprocessed_data, 'y_in_nus.npy', data=None)
	y_in_connector.read_numpys()
	y_in_ = y_in_connector.read_file

	y_out_connector = DataConnector(preprocessed_data, 'y_out_nus.npy', data=None)
	y_out_connector.read_numpys()
	y_out_ = y_out_connector.read_file

	sequences_processing = SequenceProcessing(indices_words=None, words_indices=None, encoder_length=encoder_length, decoder_length=decoder_length)
	doc_pair, x_pair, y_pair_in, y_pair_out = sequences_processing.pairing_data_(X, y_in_, y_out_)

	print("\nshape of x_pair in training set: %s\n"%str(np.array(x_pair).shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(np.array(y_pair_in).shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(np.array(y_pair_out).shape))

	x_pair_, y_pair_in_, y_pair_out_ = sequences_processing.pad_sequences(encoder_length, decoder_length, x_pair, y_pair_in, y_pair_out)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_.shape))
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_in_.shape))
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_out_.shape))
	
	doc_in_connector = DataConnector(preprocessed_data, 'doc_pair_nus.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(preprocessed_data, 'x_pair_nus.npy', x_pair_)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(preprocessed_data, 'y_pair_in_nus.npy', y_pair_in_)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(preprocessed_data, 'y_pair_out_nus.npy', y_pair_out_)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_train(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing training data to train the model...")
	sys.stdout.flush()

	x_train_connector = DataConnector(data_path, 'X_train_nus.npy', data=None)
	x_train_connector.read_numpys()
	X_train = x_train_connector.read_file

	y_train_in_connector = DataConnector(data_path, 'y_train_in_nus.npy', data=None)
	y_train_in_connector.read_numpys()
	y_train_in = y_train_in_connector.read_file

	y_train_out_connector = DataConnector(data_path, 'y_train_out_nus.npy', data=None)
	y_train_out_connector.read_numpys()
	y_train_out = y_train_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair_train_, y_pair_train_in_, y_pair_train_out_ = sequences_processing.pairing_data_(x_in=X_train, y_in=y_train_in, y_out=y_train_out)

	x_pair_train, y_pair_train_in, y_pair_train_out = sequences_processing.pad_sequences(max_len_enc=encoder_length, max_len_dec=decoder_length, in_enc_sequences=x_pair_, in_dec_sequences=y_pair_in_, out_dec_sequences=y_pair_out_)


	print("\nshape of x_pair in training set: %s\n"%str(x_pair_train.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in training set: %s\n"%str(y_pair_train_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in training set: %s\n"%str(y_pair_train_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(data_path, 'doc_pair_train_nus.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_train_nus.npy', x_pair_train)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_train_in_nus.npy', y_pair_train_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_train_out_nus.npy', y_pair_train_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()


def pair_test(params):

	data_path = params['data_path']

	print("\n=========\n")
	sys.stdout.flush()

	print(str(datetime.now()))
	sys.stdout.flush()

	t0 = time.time()

	print("Pairing test data to train the model...")
	sys.stdout.flush()

	X_test_connector = DataConnector(data_path, 'X_test_nus.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(data_path, 'y_test_in_nus.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(data_path, 'y_test_out_nus.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	sequences_processing = SequenceProcessing()
	doc_pair, x_pair_test_, y_pair_test_in_, y_pair_test_out_ = sequences_processing.pairing_data_(x_in=X_test, y_in=y_test_in, y_out=y_test_out)

	x_pair_test, y_pair_test_in, y_pair_test_out = sequences_processing.pad_sequences(max_len_enc=encoder_length, max_len_dec=decoder_length, in_enc_sequences=x_pair_test_, in_dec_sequences=y_test_in, out_dec_sequences=y_test_out)


	print("\nshape of x_pair in test set: %s\n"%str(x_pair_test.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_in in test set: %s\n"%str(y_pair_test_in.shape))
	sys.stdout.flush()
	print("\nshape of y_pair_out in test set: %s\n"%str(y_pair_test_out.shape))
	sys.stdout.flush()

	doc_in_connector = DataConnector(data_path, 'doc_pair_test_nus.npy', doc_pair)
	doc_in_connector.save_numpys()
	x_in_connector = DataConnector(data_path, 'x_pair_test_nus.npy', x_pair_test)
	x_in_connector.save_numpys()
	y_in_connector = DataConnector(data_path, 'y_pair_test_in_nus.npy', y_pair_test_in)
	y_in_connector.save_numpys()
	y_out_connector = DataConnector(data_path, 'y_pair_test_out_nus.npy', y_pair_test_out)
	y_out_connector.save_numpys()

	t1 = time.time()
	print("Pairing training set into sequences of inputs - outputs done in %.3fsec" % (t1 - t0))
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

	kp_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	kp_connector.read_numpys()
	all_keyphrases = kp_connector.read_file

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

	kp_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	kp_connector.read_numpys()
	all_keyphrases = kp_connector.read_file

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

	kp_connector = DataConnector(data_path, 'output_tokens.npy', data=None)
	kp_connector.read_numpys()
	kps = kp_connector.read_file

	in_connector = DataConnector(data_path, 'input_tokens.npy', data=None)
	in_connector.read_numpys()
	in_tokens = in_connector.read_file

	compute_presence_ = SequenceProcessing()
	all_npresence_, all_nabsence_ = compute_presence_.compute_presence(in_tokens, kps)
	total = np.sum(all_npresence_) + np.sum(all_nabsence_)

	persen_absence = np.sum(all_nabsence_) / total
	persen_presence = np.sum(all_npresence_) / total

	print(" Absent key phrase: %s" %persen_absence)
	sys.stdout.flush()
	print(" Present key phrase: %s" %persen_presence)
	sys.stdout.flush()

	print(" all_npresence_[:10]: %s" %(all_npresence_[:10]))
	sys.stdout.flush()
	print(" all_nabsence_[:10]: %s" %(all_nabsence_[:10]))
	sys.stdout.flush()
	


	t1 = time.time()
	print("Computing presence or absence of key phrases per document done in %.3fsec" % (t1 - t0))
	sys.stdout.flush()