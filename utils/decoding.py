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
from datetime import datetime
import time
import _pickle as cPickle
from utils.beam_tree import Node


class Decoding():

	# input here is individual text
	def __init__(self, encoder_model, decoder_model, indices_words, words_indices, enc_in_seq, labels, decoder_length, rnn_dim,  beam_width, num_hypotheses, filepath, filename):

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
		self.filename = filename
		self.beam_width = beam_width
		self.num_hypotheses = num_hypotheses
		self.decoded_keyphrase = ''
		self.hypotheses = []
		
	

	def greedy_decoder(self, texts, labels):

		def save_(data, filename, filepath):


			f = open(os.path.join(filepath, filename), 'wb')
			cPickle.dump(data, f)
			f.close()

			print(" file saved to: %s"%(os.path.join(filepath, filename)))

		self.enc_in_seq = texts
		self.labels = labels

		all_greedy_keyphrases = [] 
		for i, keyphrases in enumerate(self.greedy_generator()):

			print("Start greedy decoding: data-%s"%i)
			sys.stdout.flush()

			save_(data=keyphrases, filename='keyphrases-greedy-%s-%s'%(self.filename, i), filepath=self.filepath)
			all_greedy_keyphrases.append(keyphrases)

		return all_greedy_keyphrases


	def greedy_generator(self):

		for i in range(len(self.enc_in_seq)):

	
			text = self.enc_in_seq[i:i+1]
			labels = self.labels[i:i+1]

			# Encode the input as state vectors.
			states_value = self.encoder_model.predict(text)
			
			target_seq = np.zeros((1, 1))
			target_seq[0][0] = int(self.words_indices['<start>'])
			
			stop_condition = False
			start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]
			decoded_int = []
			cnt_token = 0
			
			for t_ in range(self.decoder_length+1):

				print("time step: %s" % (t_))
				sys.stdout.flush()

				t0 = time.time()

				label_t = labels[:,t_,:]
			
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

				t1 = time.time()
				print("Greedy search decoding is done in %.3fsec for time step-%s" % ((t1 - t0), t_))
				sys.stdout.flush()

			decoded_keyphrase = [self.indices_words[idx] for idx in decoded_int if idx not in start_end_idx]
				
			self.decoded_keyphrase = ' '.join(decoded_keyphrase) 
			
			yield self.decoded_keyphrase

	'''
	store list of generated key phrase per document as pickle file 
	and return all generated key phrase in test set

	'''

	def beam_decoder(self, texts, labels):

		def save_(data, filename, filepath):


			f = open(os.path.join(filepath, filename), 'wb')
			cPickle.dump(data, f)
			f.close()

			print(" file saved to: %s"%(os.path.join(filepath, filename)))

		self.enc_in_seq = texts
		self.labels = labels

		all_beam_keyphrases = []
		for i, keyphrases in enumerate(self.beam_generator()):

			print("Start beam decoding: data-%s"%i)
			sys.stdout.flush()

			save_(data=keyphrases, filename='keyphrases-beam-%s-%s'%(self.filename, i), filepath=self.filepath)
			all_beam_keyphrases.append(keyphrases)

		return all_beam_keyphrases

	'''
	N-beam search decoding
	yield (return) list of generated key phrases per document
	based on N-top of beam width  
	'''
	def beam_generator(self):

		for i in range(len(self.enc_in_seq)):


			text = self.enc_in_seq[i:i+1]
			labels = self.labels[i:i+1]

			states_value = self.encoder_model.predict(text)
			start_id = self.words_indices['<start>']

			next_fringe = [Node(parent=None, state=states_value, value=start_id, cost=0.0, extras=None)]
			hypotheses = []
			

			for t_ in range(self.decoder_length+1):

				print("time step: %s" % (t_))
				sys.stdout.flush()

				t0 = time.time()
				
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

					label_t = labels[:, t_, :]
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

				t1 = time.time()
				print("Beam search decoding is done in %.3fsec for time step-%s" % ((t1 - t0), t_))
				sys.stdout.flush()

			hypotheses.sort(key=lambda n: n.cum_cost)

			self.hypotheses = hypotheses[:self.num_hypotheses]

			yield self.hypotheses

	