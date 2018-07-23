# -*- coding: utf-8 -*-
# author: @inimah
# date: 03.07.2018
# Module for extracting n-grams from texxt source as pair comparison of generated or gold standard keyphrases

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


class Extract():

	def __init__(self, in_texts):

		self.in_texts = in_texts
		self.n_gram = 2


	# wordlist is unigrams excluding stopwords
	def get_ngrams(self, n):

		self.n_gram = n

		all_ngrams = []
		for ngrams_tokens in self.in_ngrams_generator():
			all_ngrams.append(ngrams_tokens)

		return np.array(all_ngrams)

	def in_ngrams_generator(self):

		for text in self.in_texts:

			ngrams_list = [text[i:i+self.n_gram] for i in range(len(text)-(self.n_gram-1))]

			yield ngrams_list