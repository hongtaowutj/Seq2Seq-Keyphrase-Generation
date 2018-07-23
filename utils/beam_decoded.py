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


class BeamDecoded():

	# input here is individual text
	def __init__(self, hypotheses, words_indices, indices_words, filepath):

		
		self.hypotheses = hypotheses
		self.words_indices = words_indices
		self.indices_words = indices_words
		self.filepath = filepath
		self.keyphrases_indices = [] 
		self.keyphrases_tokens = []
		self.keyphrases = []
		self.att_weights = []


	def print_hypotheses(self):

		start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]

		i = 0
		for hypothesis in self.hypotheses:

			generated_indices = hypothesis.to_sequence_of_values()
			retrieved_idx = [idx for idx in generated_indices if idx not in start_end_idx]

			#print("generated indices: %s"%retrieved_idx)
			generated_keyphrases = [self.indices_words[i] for i in retrieved_idx]
			txt = " ".join(generated_keyphrases)
			print("generated key phrases - %s: %s"%(str(i+1), txt))
			i += 1

	# retrieve generated key phrases and attention weights
	def get_(self):

		
		pred_keyphrases_indices = []
		pred_keyphrases_tokens = []
		pred_keyphrases = []
		att_weights = []

		for hypothesis in self.hypotheses:

			generated_indices = hypothesis.to_sequence_of_values()
			att_weight = hypothesis.to_sequence_of_extras()

			pred_keyphrases_indices.append(generated_indices)
			att_weights.append(att_weight)

			generated_keyphrases = [self.indices_words[i] for i in generated_indices]
			txt = " ".join(generated_keyphrases)
			pred_keyphrases_tokens.append(generated_keyphrases)
			pred_keyphrases.append(txt)

		self.keyphrases_indices = pred_keyphrases_indices
		self.keyphrases_tokens = pred_keyphrases_tokens
		self.keyphrases = pred_keyphrases
		self.att_weights = att_weights

		return self.keyphrases_tokens, self.att_weights

	def get_hypotheses(self):

		start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]
		pred_keyphrases_indices = []
		pred_keyphrases_tokens = []
		pred_keyphrases = []

		for hypothesis in self.hypotheses:

			generated_indices = hypothesis.to_sequence_of_values()
			retrieved_idx = [idx for idx in generated_indices if idx not in start_end_idx]
			pred_keyphrases_indices.append(retrieved_idx)

			generated_keyphrases = [self.indices_words[i] for i in retrieved_idx]
			txt = " ".join(generated_keyphrases)
			pred_keyphrases_tokens.append(generated_keyphrases)
			pred_keyphrases.append(txt)

		self.keyphrases_indices = pred_keyphrases_indices
		self.keyphrases_tokens = pred_keyphrases_tokens
		self.keyphrases = pred_keyphrases

		return self.keyphrases_indices, self.keyphrases_tokens, self.keyphrases

	def decript_(self):

				
		generated_indices = self.hypotheses.to_sequence_of_values()
		retrieved_idx = [idx for idx in generated_indices]

		generated_keyphrases = [self.indices_words[i] for i in retrieved_idx]
		text_keyphrases = " ".join(generated_keyphrases)

		return text_keyphrases

	def decript_hypotheses(self):

		start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]
		
		generated_indices = self.hypotheses.to_sequence_of_values()
		retrieved_idx = [idx for idx in generated_indices if idx not in start_end_idx]

		generated_keyphrases = [self.indices_words[i] for i in retrieved_idx]
		text_keyphrases = " ".join(generated_keyphrases)

		return text_keyphrases

	 

	def decript_keyphrase_gen(self):

		start_end_idx = [int(self.words_indices['<start>']), int(self.words_indices['<end>']), int(self.words_indices['<pad>'])]

		for doc in self.hypotheses:

			kps = []
			for keyphrase in doc:

				generated_indices = keyphrase.to_sequence_of_values()
				retrieved_idx = [idx for idx in generated_indices if idx not in start_end_idx]

				generated_keyphrases = [self.indices_words[i] for i in retrieved_idx]
				text_keyphrases = " ".join(generated_keyphrases)

				kps.append(text_keyphrases)

			yield kps

