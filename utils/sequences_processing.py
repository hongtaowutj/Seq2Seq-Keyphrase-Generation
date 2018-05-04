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
	def __init__(self, indices_words, words_indices, in_texts, out_texts, encoder_length, decoder_length):

		self.in_texts = in_texts # list of tokenized texts
		self.out_texts = out_texts
		self.indices_words = indices_words
		self.words_indices = words_indices
		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
	
	'''
	for encoder part: calling sequence generator

	'''
	def intexts_to_integers(self):

		"""
		Transforms each text in texts in a sequence of integers.
		
		# Arguments
			texts: A list of texts (strings).
		# Returns
			A list of sequences.
		"""
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

		for i, text in enumerate(self.in_texts):

			print("Processing text input - %s....\n"%i)
			sys.stdout.flush()

			if len(text) > self.encoder_length:
				seq = text[:self.encoder_length]
			else:
				seq = text

			print("Original text sequence - %s : %s"%(i, seq))
			sys.stdout.flush()

			integers_vector = np.zeros((1, self.encoder_length), dtype=np.int32)

			for Tx, word in enumerate(seq):
				if word not in self.indices_words.values():
					 integers_vector[0,Tx] = self.words_indices['<unk>']
				else:
					integers_vector[0,Tx] = self.words_indices[word]

			print("Integer text sequence - %s : %s\n"%(i, integers_vector))
			sys.stdout.flush()

			yield integers_vector


	'''
	for decoder part
	'''

	def outtexts_to_integers(self):

		"""
		Transforms each text in texts in a sequence of integers.
		
		# Arguments
			texts: A list of texts (strings).
		# Returns
			A list of sequences.
		"""
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

		for i, keyphrase_list in enumerate(self.out_texts):

			print("Processing text output - %s....\n"%i)
			sys.stdout.flush()

			print("\n%s\n"%(keyphrase_list))
			sys.stdout.flush()

			keyphrase_in = []
			keyphrase_out = []

			for j, text in enumerate(keyphrase_list):

				print("\n%s\n"%(text))
				sys.stdout.flush()

				if len(text) > self.decoder_length:
					seq = text[:self.decoder_length]
				else:
					seq = text

				txt_in = list(seq)
				txt_out = list(seq)
				txt_in.insert(0,'<start>')
				txt_out.append('<end>')

				print("Original decoder input sequence - %s, %s : %s"%(i, j, txt_in))
				sys.stdout.flush()

				print("Original decoder output sequence - %s, %s : %s"%(i, j, txt_out))
				sys.stdout.flush()

				y_in = np.zeros((1, self.decoder_length+1), dtype=np.int32)
				y_out = np.zeros((1, self.decoder_length+1), dtype=np.int32)

				for Ty, word in enumerate(txt_in):
					if word not in self.indices_words.values():
						 y_in[0,Ty] = self.words_indices['<unk>']
					else:
						y_in[0,Ty] = self.words_indices[word]

				for Ty, word in enumerate(txt_out):
					if word not in self.indices_words.values():
						 y_out[0,Ty] = self.words_indices['<unk>']
					else:
						y_out[0,Ty] = self.words_indices[word]

				print("integer decoder input sequence - %s, %s : %s"%(i, j, y_in))
				sys.stdout.flush()

				print("integer decoder output sequence - %s, %s : %s"%(i, j, y_out))
				sys.stdout.flush()

				keyphrase_in.append(y_in)
				keyphrase_out.append(y_out)

			print("integer list decoder input sequences - %s : %s"%(i, keyphrase_in))
			sys.stdout.flush()

			print("integer list decoder output sequences - %s : %s"%(i, keyphrase_out))
			sys.stdout.flush()

			yield keyphrase_in, keyphrase_out

