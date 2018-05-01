# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from math import log

from utils.beam_tree import Node


class Decoding():

	# input here is individual text
	def __init__(self, encoder_model, decoder_model, indices_words, words_indices, enc_in_seq, labels, decoder_length, rnn_dim,  beam_width, filepath):

		self.encoder_model = encoder_model
		self.decoder_model = decoder_model
		self.indices_words = indices_words
		self.words_indices = words_indices
		self.enc_in_seq = enc_in_seq
		#self.dec_in_seq = dec_in_seq
		self.labels = labels
		self.decoder_length = decoder_length
		self.rnn_dim = rnn_dim
		self.filepath = filepath
		self.beam_width = beam_width
		self.num_hypotheses = beam_width
		self.decoded_keyphrase = ''
		self.hypotheses = []
		
	   

	'''
	one-best search decoding
	'''
	def decode_sampled_softmax(self):
  
		# Encode the input as state vectors.
		states_value = self.encoder_model.predict(self.enc_in_seq)
		
		target_seq = np.zeros((1, 1))
		#target_seq[0][0] = self.dec_in_seq[0][0]
		target_seq[0][0] = int(self.words_indices['<start>'])
		
		stop_condition = False
		start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]
		decoded_int = []
		cnt_token = 0
		
		for t in range(self.decoder_length+1):
		  
			label_t = self.labels[:,t,:]
		
			output_tokens, h = self.decoder_model.predict([target_seq] + [states_value] + [label_t])
			# Sample a token
			sampled_token_index = np.argmax(output_tokens)
			sampled_word = self.indices_words[sampled_token_index]
			cnt_token += 1
			decoded_int.append(sampled_token_index)
			
			# Update the target sequence (of length 1).
			target_seq = np.zeros((1, 1))
			target_seq[0][0] = sampled_token_index
			
			# Update states
			states_value = h

		decoded_keyphrase = [self.indices_words[idx] for idx in decoded_int if idx not in start_end_idx]
			
		self.decoded_keyphrase = ' '.join(decoded_keyphrase) 
		
		return self.decoded_keyphrase

	'''
	N-beam search decoding 
	'''
	def beam_search_sampled_softmax(self):

		states_value = self.encoder_model.predict(self.enc_in_seq)
		start_id = self.words_indices['<start>']

		next_fringe = [Node(parent=None, state=states_value, value=start_id, cost=0.0, extras=None)]
		hypotheses = []
		

		for t_ in range(self.decoder_length+1):
			node_next_fringe = [node.to_sequence_of_values() for node in next_fringe]
			fringe = []

			for n in next_fringe:
				if n.value == self.words_indices['<end>']:
					hypotheses.append(n) 
				else:
					fringe.append(n)
			if not fringe or len(hypotheses) >= self.num_hypotheses:
				break

			probs = []
			states = []
			predicted_indices = []

			for n in fringe:

				# Generate empty target sequence of length 1.
				dec_input = np.zeros((1, 1))

				# Populate the first character of target sequence with the start character; 
				# otherwise index of predicted token store from previous step.
				dec_input[0][0] = n.value

				# initial state = state from encoder; otherwise, state from previous cell of decoder
				states_value = n.state
				shape_states_value = str(states_value.shape)

				if states_value.shape[0] == self.rnn_dim:
					states_value = states_value.reshape((1, states_value.shape[0]))

				label_t = self.labels[:, t_, :]
				predicted_prob, state_t = self.decoder_model.predict([dec_input] + [states_value] + [label_t])

				Y_t = np.argsort(predicted_prob, axis=-1)[-self.beam_width:] # no point in taking more than fits in the beam

				#if t_ >= 1:
				#	Y_t = [Y_t[-1]]

				probs.append(predicted_prob)
				states.append(state_t)
				predicted_indices.append(Y_t)

			probs = np.array(probs)
			states = np.array(states)
			predicted_indices = np.array(predicted_indices)

			probs = probs.reshape((probs.shape[0],probs.shape[2]))
			states = states.reshape((states.shape[0],states.shape[2]))
			predicted_indices = predicted_indices.reshape((predicted_indices.shape[0],predicted_indices.shape[2]))

			# store the prediction node for each time step
			next_fringe = []
			for Y_tn, prob_tn, state_tn, n in zip(predicted_indices, probs, states, fringe):
				# turn into log domain
				logprob_tn = -np.log(prob_tn[Y_tn]) 
				# store N-top beam width index of token and its probability in tree node of tokens 
				for y_tn, logprob_tn in zip(Y_tn, logprob_tn):
					n_new = Node(parent=n, state=state_tn, value=y_tn, cost=logprob_tn, extras=None)
					next_fringe.append(n_new)
	
			next_fringe = sorted(next_fringe, key=lambda n: n.cum_cost)[:self.beam_width] # may move this into loop to save memory 

		hypotheses.sort(key=lambda n: n.cum_cost)

		self.hypotheses = hypotheses[:self.num_hypotheses]

		return self.hypotheses

	