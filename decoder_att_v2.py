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
from utils.decoding_att import Decoding
from models.seq2seq_att_sampled_softmax_v2 import AttentionSampledSoftmax


def decoder(params):

	data_path = params['data_path']
	glove_embed = params['glove_embedding']
	oov_embed = params['oov_embedding']
	preprocessed_data = params['preprocessed_data']
	decode_path = params['decode_path']
	model_path = params['model_path']
	result_path = params['result_path']
	file_name = params['file_name']
	weights = params['weights']

	encoder_length = params['encoder_length']
	decoder_length = params['decoder_length']
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
	indices_words_connector = DataConnector(preprocessed_data, 'all_idxword_vocabulary.pkl', data=None)
	indices_words_connector.read_pickle()
	indices_words = indices_words_connector.read_file

	words_indices_connector = DataConnector(preprocessed_data, 'all_wordidx_vocabulary.pkl', data=None)
	words_indices_connector.read_pickle()
	words_indices = words_indices_connector.read_file

	y_test_true_connector = DataConnector(data_path, 'test_output_tokens.npy', data=None)
	y_test_true_connector.read_numpys()
	y_test_true = y_test_true_connector.read_file


	# paired data set

	X_pair_test_connector = DataConnector(preprocessed_data, 'x_pair_test.npy', data=None)
	X_pair_test_connector.read_numpys()
	X_pair_test = X_pair_test_connector.read_file

	y_pair_test_in_connector = DataConnector(preprocessed_data, 'y_pair_test_in.npy', data=None)
	y_pair_test_in_connector.read_numpys()
	y_pair_test_in = y_pair_test_in_connector.read_file

	y_pair_test_out_connector = DataConnector(preprocessed_data, 'y_pair_test_out.npy', data=None)
	y_pair_test_out_connector.read_numpys()
	y_pair_test_out = y_pair_test_out_connector.read_file

	# non-paired data set


	X_test_connector = DataConnector(preprocessed_data, 'X_test_pad.npy', data=None)
	X_test_connector.read_numpys()
	X_test = X_test_connector.read_file

	y_test_in_connector = DataConnector(preprocessed_data, 'y_test_in.npy', data=None)
	y_test_in_connector.read_numpys()
	y_test_in = y_test_in_connector.read_file

	y_test_out_connector = DataConnector(preprocessed_data, 'y_test_out.npy', data=None)
	y_test_out_connector.read_numpys()
	y_test_out = y_test_out_connector.read_file

	print("\n Non-paired test set: \n")
	sys.stdout.flush()
	print("X (input for encoder) shape: %s"%str(X_test.shape)) # input for encoder
	sys.stdout.flush()
	print("y_in (input for decoder) shape: %s"%str(y_test_in.shape)) # input for decoder
	sys.stdout.flush()
	print("y_out (output for decoder) shape: %s\n\n"%str(y_test_out.shape)) # output for decoder
	sys.stdout.flush()



	'''
	Decoder model for inference stage
	Return: generated keyphrases
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

	sampled_softmax = AttentionSampledSoftmax(encoder_length=encoder_length, decoder_length=decoder_length, embedding_dim=embedding_dim, birnn_dim=birnn_dim, rnn_dim=rnn_dim, vocab_size=vocab_size, num_samples=num_samples, filepath=result_path, filename=file_name, batch_train_iter=None, batch_val_iter=None, batch_size=None, steps_epoch=None, val_steps=None, epochs=None)

	# skeleton of model architecture
	sampled_softmax.train_att_sampled_softmax(pretrained_embedding, oov_embedding)
	


	sampled_softmax.predict_att_sampled_softmax(weights)
	# 1. Prediction model after being trained on sampled softmax setting
	predict_softmax_model = sampled_softmax.prediction_model
	encoder_model = sampled_softmax.encoder_model

	

	
	'''

	Inference stage
	Model: layers from prediction model and decoder model
	Inference (text generation) approach: 
	1. One best search decoding (Greedy search): 
	   Return one best (top) probable word sequence, from joint probability of words within decoder time steps (decoder sequence length)
	2. N-Beam search decoding: 
	   Return N-top best most probable word sequences, by utilizing beam tree search per time steps and joint probability within decoder time steps (decoder sequence length)

	'''

	decoder_model = sampled_softmax.create_decoder_model()

	
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
	print("\nBeam width: %s\n" %beam_width)
	sys.stdout.flush()
	num_hypotheses = beam_width

	y_dummy_test = np.zeros((len(X_test), decoder_length+1, 1))
	s0_test = np.zeros((len(X_test), rnn_dim))
	att0_test = np.zeros((len(X_test),encoder_length,1))


	print(str(datetime.now()))
	sys.stdout.flush()


	inference_mode = Decoding(encoder_model=encoder_model, decoder_model=decoder_model, indices_words=indices_words, words_indices=words_indices, enc_in_seq=None, labels=None,states=None, attentions=None, decoder_length=decoder_length, rnn_dim=rnn_dim, beam_width=beam_width, num_hypotheses=num_hypotheses, filepath=decode_path, filename=file_name)

	

	t0_1 = time.time()
	print("Start beam decoding...")
	sys.stdout.flush()

	beam_keyphrases = inference_mode.beam_decoder(X_test[:500], y_dummy_test[:500], s0_test[:500], att0_test[:500])
	
	beam_decode_connector = DataConnector(decode_path, 'beam_kp-%s.npy'%(file_name), beam_keyphrases)
	beam_decode_connector.save_numpys()

	t1_1 = time.time()
	print("Beam decoding is done in %.3fsec" % (t1_1 - t0_1))
	sys.stdout.flush()

	


