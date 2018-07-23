# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import re
import nltk
import string
from string import punctuation
import json
from collections import OrderedDict
nltk.data.path.append('/home/TUE/inimah/nltk_data')
sno = nltk.stem.SnowballStemmer('english')
from nltk.corpus import stopwords
import _pickle as cPickle


class SequenceProcessing():

	# input here is individual text
	def __init__(self, indices_words, words_indices, encoder_length, decoder_length):

		self.in_texts = [] # list of tokenized texts
		self.out_texts = []
		self.max_sents = 0
		self.indices_words = indices_words
		self.words_indices = words_indices
		self.num_words = 0
		self.x_in = []
		self.y_in = []
		self.y_out = []
		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
	
	'''
	for encoder part: calling sequence generator

	'''
	def intexts_to_integers(self, in_texts):

		"""
		Transforms each text in texts in a sequence of integers.
		
		# Arguments
			texts: A list of texts (strings).
		# Returns
			A list of sequences.
		"""

		self.in_texts = in_texts # list of tokenized texts
		self.num_words = len(self.indices_words)

		res = []
		for vect in self.intexts_to_integers_generator():
			res.append(vect)
		return res

	'''
	sequence generator for encoder part

	'''

	def intexts_to_integers_generator(self):

		"""Transforms each text in `texts` in a sequence of integers.
		Each item in texts can also be a list, in which case we assume each item of that list
		to be a token.
		Only top "num_words" most frequent words will be taken into account.
		Only words known by the tokenizer will be taken into account.
		# Arguments
			texts: A list of texts (strings).
		# Yields
			Yields individual sequences.
		"""
		for text in self.in_texts:

			seq = text[:self.encoder_length]
			
			integers_vector = []

			for word in seq:
				idx = self.words_indices.get(word)
				if idx != None:
					if self.num_words and idx >= self.num_words:
						continue
					else:
						integers_vector.append(idx) 
				else:
					idx = self.words_indices.get('<unk>')
					if idx != None:
						integers_vector.append(idx) 

			yield integers_vector

	def in_sents_to_integers(self, in_texts, max_sents):

		"""
		Transforms each text in texts in a sequence of integers.
		
		# Arguments
			texts: A list of texts (strings)  splitted into sentences.
		# Returns
			A list of sequences.
		"""
		self.in_texts = in_texts # list of tokenized texts
		self.max_sents = max_sents
		self.num_words = len(self.indices_words)

		res = []
		for vect in self.in_sents_to_integers_generator():
			res.append(vect)
		return res

	def in_sents_to_integers_generator(self):

		"""Transforms each text in `texts` in a sequence of integers.
		Each item in texts can also be a list, in which case we assume each item of that list
		to be a token.
		Only top "num_words" most frequent words will be taken into account.
		Only words known by the tokenizer will be taken into account.
		# Arguments
			texts: A list of texts (strings) splitted into sentences.
		# Yields
			Yields individual sequences.
		"""
		for text in self.in_texts:

			sent_vector = []

			for j, sent in enumerate(text):

				if j < self.max_sents:

					seq = sent[:self.encoder_length]

				integers_vector = []

				for word in seq:
					idx = self.words_indices.get(word)
					if idx != None:
						if self.num_words and idx >= self.num_words:
							continue
						else:
							integers_vector.append(idx) 
					else:
						idx = self.words_indices.get('<unk>')
						if idx != None:
							integers_vector.append(idx)

				sent_vector.append(integers_vector)

			yield sent_vector

	'''
	for decoder part
	'''

	def outtexts_to_integers(self, out_texts):

		"""
		Transforms each text in texts in a sequence of integers.
		
		# Arguments
			texts: A list of texts (strings).
		# Returns
			A list of sequences.
		"""

		self.out_texts = out_texts
		self.num_words = len(self.indices_words)

		res_in = []
		res_out = []
		
		for (keyphrase_in, keyphrase_out) in self.outtexts_to_integers_generator():
			res_in.append(keyphrase_in)
			res_out.append(keyphrase_out)

		return res_in, res_out


	def outtexts_to_integers_generator(self):

		"""Transforms each text in `texts` in a sequence of integers.
		Each item in texts can also be a list, in which case we assume each item of that list
		to be a token.
		Only top "num_words" most frequent words will be taken into account.
		Only words known by the tokenizer will be taken into account.
		# Arguments
			texts: A list of texts (strings).
		# Yields
			Yields individual sequences.
		"""


		for keyphrase_list in self.out_texts:

			keyphrase_in = []
			keyphrase_out = []

			for text in keyphrase_list:

				seq = text[:self.decoder_length]
				
				txt_in = list(seq)
				txt_out = list(seq)
				txt_in.insert(0,'<start>')
				txt_out.append('<end>')

				integers_vector_in = []
				integers_vector_out = []

				for word in txt_in:

					idx =  self.words_indices.get(word)
					if idx != None:
						if self.num_words and idx >= self.num_words:
							continue
						else:
							integers_vector_in.append(idx) 
					else:
						idx = self.words_indices.get('<unk>')
						if idx != None:
							integers_vector_in.append(idx)


				for word in txt_out:

					idx = self.words_indices.get(word)
					if idx != None:
						if self.num_words and idx >= self.num_words:
							continue
						else:
							integers_vector_out.append(idx) 
					else:
						idx = self.words_indices.get('<unk>')
						if idx != None:
							integers_vector_out.append(idx)

				keyphrase_in.append(integers_vector_in)
				keyphrase_out.append(integers_vector_out)

			yield keyphrase_in, keyphrase_out

	def pad_sequences_sent_in(self, max_len, max_sents, sequences):

		x = (np.ones((len(sequences), max_sents, max_len)) * 0.).astype('int32')
		for idx, text in enumerate(sequences):

			for sentid, seq in enumerate(text):

				if sentid < max_sents:

					# skip empty array, if exists
					if not len(seq):
						continue

					seq = seq[:max_len]
					x[idx,sentid,:len(seq)] = seq
				
		return x

	def pad_sequences_in(self, max_len, sequences):

		x = (np.ones((len(sequences), max_len)) * 0.).astype('int32')
		for idx, seq in enumerate(sequences):

			# skip empty array, if exists
			if not len(seq):
				continue

			seq = seq[:max_len]
			x[idx,:len(seq)] = seq
			
		return x

	'''
	Do padding after pairing sequences into one text sequence and one target sequence
	for paired inputs - outputs
	'''
	def pad_sequences(self, max_len_enc, max_len_dec, in_enc_sequences, in_dec_sequences, out_dec_sequences):

		self.encoder_length = max_len_enc
		self.decoder_length = max_len_dec

		print("in_enc_sequences shape: %s"%str(in_enc_sequences.shape))

		x = (np.ones((len(in_enc_sequences), self.encoder_length)) * 0.).astype('int32')
		y_in = (np.ones((len(in_dec_sequences), self.decoder_length+1)) * 0.).astype('int32')
		y_out = (np.ones((len(out_dec_sequences), self.decoder_length+1)) * 0.).astype('int32')

		for idx in range(len(in_enc_sequences)):

			# skip empty array, if exists
			if not len(in_enc_sequences[idx]):
				continue
			if not len(in_dec_sequences[idx]):
				continue
			if not len(out_dec_sequences[idx]):
				continue

			seq_enc_in = in_enc_sequences[idx]
			seq_enc_in = seq_enc_in[:self.encoder_length]

			seq_dec_in = in_dec_sequences[idx]
			seq_dec_in = seq_dec_in[:self.decoder_length+1]

			seq_dec_out = out_dec_sequences[idx]
			seq_dec_out = seq_dec_out[:self.decoder_length+1]

			x[idx,:len(seq_enc_in)] = seq_enc_in
			y_in[idx,:len(seq_dec_in)] = seq_dec_in
			y_out[idx,:len(seq_dec_out)] = seq_dec_out

		return x, y_in, y_out


	# NOT BEING USED

	def pairing_data(self, x_in=None, y_in=None, y_out=None):

		self.x_in = x_in
		self.y_in = y_in
		self.y_out = y_out

		docid_pair = []
		x_pair = []
		y_pair_in = []
		y_pair_out = []

		for (docid_pair_, x_pair_, y_pair_in_, y_pair_out_) in self.pair_generator():
			docid_pair.append(docid_pair_)
			x_pair.append(x_pair_)
			y_pair_in.append(y_pair_in_)
			y_pair_out.append(y_pair_out_)

		return docid_pair, x_pair, y_pair_in, y_pair_out

	# NOT BEING USED

	def pair_generator(self):

		for i, (y_in_, y_out_) in enumerate(zip(self.y_in, self.y_out)):
			

			for j in range(len(y_in_)):

				docid_pair = i
				x_pair = self.x_in[i]
				y_pair_in = y_in_[j]
				y_pair_out = y_out_[j]

				yield docid_pair, x_pair, y_pair_in, y_pair_out

	# USE THIS
	def pairing_data_(self, x_in, y_in, y_out):

		self.x_in = x_in
		self.y_in = y_in
		self.y_out = y_out

		docid_pair = []
		y_pair_in = []
		y_pair_out = []

		# count number of key phrases per document
		len_y = []
		for y_in_list in self.y_in:
			len_y.append(len(y_in_list))


		x_pair = np.repeat(np.array(self.x_in), len_y, axis=0)

		for (docid_pair_, x_pair_, y_pair_in_, y_pair_out_) in self.pair_generator():
			docid_pair.append(docid_pair_)
			y_pair_in.append(y_pair_in_)
			y_pair_out.append(y_pair_out_)

		print("x_pair shape: %s"%str(x_pair.shape))

		return docid_pair, x_pair, y_pair_in, y_pair_out

	# USE THIS
	def pair_generator_(self):

		for i, (y_in_, y_out_) in enumerate(zip(self.y_in, self.y_out)):
		
			for j in range(len(y_in_)):

				docid_pair = i
				y_pair_in = y_in_[j]
				y_pair_out = y_out_[j]

				yield docid_pair, y_pair_in, y_pair_out

	def compute_presence_gen(self):



		for i in range(len(self.out_texts)):

			out_texts_i = sum(self.out_texts[i], [])

			n_outtext = len(set(out_texts_i))

			presence_list = set(out_texts_i) & set(self.in_texts[i])
			n_presence = len(presence_list)

			absence_list = set(out_texts_i) - set(self.in_texts[i])
			n_absence = len(absence_list)

			if i<5:
				print("n_outtext: %s"%n_outtext)
				print("n_presence: %s"%n_presence)
				print("n_absence: %s"%n_absence)

			yield n_presence, n_absence

	def compute_presence(self, input_tokens=None, output_tokens=None):

		self.in_texts = input_tokens
		self.out_texts = output_tokens

		all_npresence = []
		all_nabsence = []

		for (n_presence, n_absence) in self.compute_presence_gen():

			all_npresence.append(n_presence)
			all_nabsence.append(n_absence)

		return all_npresence, all_nabsence





