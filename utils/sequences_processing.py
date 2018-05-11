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
	def __init__(self, indices_words=None, words_indices=None, encoder_length=None, decoder_length=None):

		self.in_texts = None # list of tokenized texts
		self.out_texts = None
		self.indices_words = indices_words
		self.words_indices = words_indices
		self.num_words = 0
		self.x_in = None
		self.y_in = None
		self.y_out = None
		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
	
	'''
	for encoder part: calling sequence generator

	'''
	def intexts_to_integers(self, in_texts = None):

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


			if len(text) > self.encoder_length:
				seq = text[:self.encoder_length]
			else:
				seq = text

			integers_vector = []

			for word in seq:
				idx = self.words_indices[word]
				if idx != None:
					if self.num_words and idx >= self.num_words:
						continue
					else:
						integers_vector.append(idx) 
				else:
					idx = self.words_indices['<unk>']
					if idx != None:
						integers_vector.append(idx) 

			yield integers_vector


	'''
	for decoder part
	'''

	def outtexts_to_integers(self, out_texts = None):

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

				if len(text) > self.decoder_length:
					seq = text[:self.decoder_length]
				else:
					seq = text

				txt_in = list(seq)
				txt_out = list(seq)
				txt_in.insert(0,'<start>')
				txt_out.append('<end>')

				integers_vector_in = []
				integers_vector_out = []

				for word in txt_in:

					idx =  self.words_indices[word]
					if idx != None:
						if self.num_words and idx >= self.num_words:
							continue
						else:
							integers_vector_in.append(idx) 
					else:
						idx = self.words_indices['<unk>']
						if idx != None:
							integers_vector_in.append(idx)


				for word in txt_out:

					idx = self.words_indices[word]
					if idx != None:
						if self.num_words and idx >= self.num_words:
							continue
						else:
							integers_vector_out.append(idx) 
					else:
						idx = self.words_indices['<unk>']
						if idx != None:
							integers_vector_out.append(idx)

				keyphrase_in.append(integers_vector_in)
				keyphrase_out.append(integers_vector_out)

			yield keyphrase_in, keyphrase_out

	
	def pairing_data_(self, x_in=None, y_in=None, y_out=None):

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

		x_pair = np.repeat(self.x_in, len_y, axis=0)

		for (docid_pair_, x_pair_, y_pair_in_, y_pair_out_) in self.pair_generator():
			docid_pair.append(docid_pair_)
			y_pair_in.append(y_pair_in_)
			y_pair_out.append(y_pair_out_)

		return docid_pair, x_pair, y_pair_in, y_pair_out

	
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





