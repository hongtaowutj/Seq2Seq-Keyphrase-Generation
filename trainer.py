import os
import sys
sys.path.append(os.getcwd())

import tensorflow as tf
from keras.models import Model
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical

from utils.data_connector import DataConnector
from utils.true_keyphrases import TrueKeyphrases
from utils.decoding import Decoding
from models.seq2seq_sampled_softmax import SampledSoftmax


def trainer(params):

	try:

		data_path = params['data_path']
		model_path = params['model_path']
		result_path = params['result_path']

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
		indices_words_connector = DataConnector(data_path, 'indices_words.pkl', data=None)
		indices_words_connector.read_pickle()
		indices_words = indices_words_connector.read_file

		words_indices_connector = DataConnector(data_path, 'words_indices.pkl', data=None)
		words_indices_connector.read_pickle()
		words_indices = words_indices_connector.read_file


		'''
		Reading X, y pair data set for training and validating model

		'''
		# 1. training set

		X_train_connector = DataConnector(data_path, 'x_pair_train.pkl', data=None)
		X_train_connector.read_pickle()
		X_train = X_train_connector.read_file

		y_train_in_connector = DataConnector(data_path, 'y_pair_train_in.pkl', data=None)
		y_train_in_connector.read_pickle()
		y_train_in = y_train_in_connector.read_file

		y_train_out_connector = DataConnector(data_path, 'y_pair_train_out.pkl', data=None)
		y_train_out_connector.read_pickle()
		y_train_out = y_train_out_connector.read_file

		print("\n X,y pair of training set: \n")
		print("X (input for encoder) shape: %s"%str(X_train.shape)) # input for encoder
		print("y_in (input for decoder) shape: %s"%str(y_train_in.shape)) # input for decoder
		print("y_out (output for decoder) shape: %s\n\n"%str(y_train_out.shape)) # output for decoder

		# 2. validation set

		# pair data set

		X_valid_pair_connector = DataConnector(data_path, 'x_pair_valid.pkl', data=None)
		X_valid_pair_connector.read_pickle()
		X_valid_pair = X_valid_pair_connector.read_file

		y_valid_in_pair_connector = DataConnector(data_path, 'y_pair_valid_in.pkl', data=None)
		y_valid_in_pair_connector.read_pickle()
		y_valid_in_pair = y_valid_in_pair_connector.read_file

		y_valid_out_pair_connector = DataConnector(data_path, 'y_pair_valid_out.pkl', data=None)
		y_valid_out_pair_connector.read_pickle()
		y_valid_out_pair = y_valid_out_pair_connector.read_file

		print("\n X, y pair of validation set: \n")
		print("X (input for encoder) shape: %s"%str(X_valid_pair.shape)) # input for encoder
		print("y_in (input for decoder) shape: %s"%str(y_valid_in_pair.shape)) # input for decoder
		print("y_out (output for decoder) shape: %s\n\n"%str(y_valid_out_pair.shape)) # output for decoder

		# non-pair data set

		X_valid_connector = DataConnector(data_path, 'X_valid.pkl', data=None)
		X_valid_connector.read_pickle()
		X_valid = X_valid_connector.read_file

		y_valid_in_connector = DataConnector(data_path, 'y_valid_in.pkl', data=None)
		y_valid_in_connector.read_pickle()
		y_valid_in = y_valid_in_connector.read_file

		y_valid_out_connector = DataConnector(data_path, 'y_valid_out.pkl', data=None)
		y_valid_out_connector.read_pickle()
		y_valid_out = y_valid_out_connector.read_file

		print("\n Non-paired validation set: \n")
		print("X (input for encoder) shape: %s"%str(X_valid.shape)) # input for encoder
		print("y_in (input for decoder) shape: %s"%str(y_valid_in.shape)) # input for decoder
		print("y_out (output for decoder) shape: %s\n\n"%str(y_valid_out.shape)) # output for decoder


		# 3. test set

		X_test_connector = DataConnector(data_path, 'X_test.pkl', data=None)
		X_test_connector.read_pickle()
		X_test = X_test_connector.read_file

		y_test_in_connector = DataConnector(data_path, 'y_test_in.pkl', data=None)
		y_test_in_connector.read_pickle()
		y_test_in = y_test_in_connector.read_file

		y_test_out_connector = DataConnector(data_path, 'y_test_out.pkl', data=None)
		y_test_out_connector.read_pickle()
		y_test_out = y_test_out_connector.read_file

		print("\n Non-paired test set: \n")
		print("X (input for encoder) shape: %s"%str(X_test.shape)) # input for encoder
		print("y_in (input for decoder) shape: %s"%str(y_test_in.shape)) # input for decoder
		print("y_out (output for decoder) shape: %s\n\n"%str(y_test_out.shape)) # output for decoder


		# 4. y_true (true keyphrases) from test set

		y_test_true_connector = DataConnector(data_path, 'test_output_tokens.pkl', data=None)
		y_test_true_connector.read_pickle()
		y_test_true = y_test_true_connector.read_file

		'''
		Data iterator: preparing per batch training set

		'''

		steps_epoch = len(X_train)/batch_size
		batch_iter = Dataiterator(X_train, y_train_in, y_train_out, decoder_dim=rnn_dim, batch_size=batch_size)


	except:
		raise

	'''
	1. Initiate model for training Seq2Seq with sampled softmax layer
	2. Compile with sampled softmax training loss, as an underestimate of full softmax loss
	3. Train with per-batch samples

	'''

	sampled_softmax = SampledSoftmax(encoder_length, decoder_length, embedding_dim, birnn_dim, rnn_dim, vocab_size, num_samples, result_path, batch_iter, batch_size, steps_epoch, epoch)

	'''
	Model for evaluating sampled softmax layer on full softmax
	Return: loss of trained model in sampled softmax (an underestimate of full softmax)
	'''

	sampled_softmax.train_sampled_softmax()
	sampled_softmax.compile_()
	sampled_softmax.train_()
	sampled_softmax.plot_()

	# get stored trained model and layers
	ssoftmax_train_model = sampled_softmax.train_model

	# get encoder model and corresponding layers
	# 1. encoder model
	encoder_model = sampled_softmax.encoder_model
	# 2. input layer of encoder 
	in_encoder = sampled_softmax.in_encoder
	# 3. output layer of decoder
	out_bidir_encoder = sampled_softmax.out_bidir_encoder

	
	# get corresponding layers for decoder model
	# 1. forward GRU decoder layer 
	fwd_decoder = sampled_softmax.fwd_decoder
	# 2. input layer of decoder
	in_decoder = sampled_softmax.in_decoder
	# 3. embedding layer of decoder
	embed_decoder = sampled_softmax.embed_decoder
	# 4. output of embedding layer of decoder
	in_dec_embedded = sampled_softmax.in_dec_embedded


	# label layer as input of sampling softmax layer
	labels = sampled_softmax.labels

	'''
	Model for evaluating sampled softmax layer on full softmax
	Return: loss of trained model in full softmax setting
	'''

	sampled_softmax.eval_sampled_softmax()
	eval_softmax_model = sampled_softmax.eval_model

	'''
	Model for retrieving perplexity loss
	Return: perplexity loss of trained model
	'''

	sampled_softmax.perplexity_loss()
	perplexity_model = sampled_softmax.perplexity_model

	'''
	Model for retrieving softmax probability
	Return: softmax probability of prediction layer
	'''

	sampled_softmax.predict_sampled_softmax()
	# 1. Prediction model after being trained on sampled softmax setting
	predict_softmax_model = sampled_softmax.prediction_model
	# 2. Prediction layer
	pred_softmax = sampled_softmax.pred_softmax


	'''
	Decoder model for inference stage
	Return: generated keyphrases
	'''

	sampled_softmax.create_decoder_model()
	decoder_model = sampled_softmax.decoder_model


	'''
	Compute softmax loss on validation set
	Model: 'Eval' mode sampled softmax
	Return: Loss on validation set
	'''

	y_dec_valid_out = to_categorical(y_valid_out, vocab_size)
	# reshape y to 3D dimension (batch_size, sequence_length, 1)
	y_true_valid = y_valid_out.reshape((y_valid_out.shape[0], y_valid_out.shape[1], 1))
	outputs_valid = list(y_dec_valid_out.swapaxes(0,1))
	m_valid = X_valid.shape[0]
	s0_valid = np.zeros((m_valid, rnn_dim))

	score = eval_softmax_model.evaluate([X_valid, y_valid_in, s0_valid, y_true_valid], outputs_valid, batch_size=64)
	print("average loss: %s"%str(score[0]/(decoder_length+1)))
	print("all time steps loss: %s"%score)


	'''
	Compute perplexity loss on validation set
	Model: 'Eval' mode sampled softmax
	Return: Perplexity loss on validation set
	'''

	perplex_score = perplexity_model.evaluate([X_valid, y_valid_in, s0_valid, y_true_valid], outputs_valid, batch_size=64)
	print("average perplexity score: %s"%str(perplex_score[0]/(decoder_length+1)))
	print("all time steps perplexity score: %s"%perplex_score)


	'''

	Inference stage
	Model: layers from prediction model and decoder model
	Inference (text generation) approach: 
	1. One best search decoding (Greedy search): 
	   Return one best (top) probable word sequence, from joint probability of words within decoder time steps (decoder sequence length)
	2. N-Beam search decoding: 
	   Return N-top best most probable word sequences, by utilizing beam tree search per time steps and joint probability within decoder time steps (decoder sequence length)

	'''

	
	# transform tokenized y_true (ground truth of keyphrases) into full sentences / keyphrases
	keyphrases_transform =  TrueKeyphrases(y_test_true)
	keyphrases_transform.get_true_keyphrases()
	keyphrases_transform.get_stat_keyphrases()
	y_true = keyphrases_transform.y_true
	max_kp_num = keyphrases_transform.max_kp_num
	mean_kp_num = keyphrases_transform.mean_kp_num
	std_kp_num = keyphrases_transform.std_kp_num

	print("Maximum number of key phrases per document in corpus: %s" %max_kp_num)
	print("Average number of key phrases per document in corpus: %s" %mean_kp_num)
	print("Standard Deviation of number of key phrases per document in corpus: %s" %std_kp_num)

	# round up function for computing beam width 
	def roundup(x):
		return x if x % 5 == 0 else x + 5 - x % 5

	beam_width = int(roundup(mean_kp_num + (3 * std_kp_num)))
	print("Beam width: %s" %beam_width)

	# reshape y to 3D dimension (batch_size, sequence_length, 1)
	y_test = y_test_out.reshape((y_test_out.shape[0], y_test_out.shape[1], 1)) # as true labels

	all_greedy_keyphrases = [] # for storing the results from greedy search approach
	all_beam_keyphrases = [] # for storing the results from beam search approach

	for i in range(len(X_test)):

		inference_mode = Decoding(encoder_model, decoder_model, indices_words, words_indices, X_test[i:i+1], y_test_in[i:i+1], y_test[i:i+1], decoder_length, rnn_dim, beam_width, result_path)

		# One best search (Greedy search)
		inference_mode.decode_sampled_softmax()
		decoded_keyphrase = inference_mode.decoded_keyphrase
		all_greedy_keyphrases.append(decoded_keyphrase)

		# N- beam search
		inference_mode.beam_search_sampled_softmax()
		hypotheses = inference_mode.hypotheses
		all_beam_keyphrases.append(hypotheses)

	greedy_decode_connector = DataConnector(result_path, 'all_greedy_keyphrases.pkl', all_greedy_keyphrases)
	greedy_decode_connector.save_pickle()

	beam_decode_connector = DataConnector(result_path, 'all_beam_keyphrases.pkl', all_beam_keyphrases)
	beam_decode_connector.save_pickle()





	




