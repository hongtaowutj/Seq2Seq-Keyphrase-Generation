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


class Preprocessing():

	# input here is individual text
	def __init__(self, input_text=None, output_keyphrases=None):

		self.inputs = input_text
		self.outputs = output_keyphrases
		kp_unlisted_punct = ['-', '_', '+', '#']
		self.kp_punct = ''.join([p for p in string.punctuation if p not in kp_unlisted_punct])
		eof_punc = ['.','?','!',',',';','-','_','+', '#']
		self.input_punct = [x for x in string.punctuation if x not in eof_punc]
		self.prep_inputs = []
		self.prep_outputs = []
		self.inputs_tokens = [] 
		self.output_tokens = []
		self.all_tokens = []

	# generator for cleaning raw data
	# input is list of text inputs and text outputs
	def preprocess_in_generator(self):

		for text_in in self.inputs:

			text_in = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', text_in)
			text_in = re.sub(r'/', ' / ', text_in) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
			text_in = re.sub(r'@\w+', '<USER>', text_in)
			text_in = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' ', text_in) # eliminate numbers
			text_in = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', text_in) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			text_in = re.sub(r'[^\x00-\x7f]', ' ', text_in) # encoded characters
			punct_list = str.maketrans({key: None for key in self.input_punct})
			text_in = text_in.translate(punct_list)
			text_in = re.sub(r'[\-\_\.\?\:\;]+\ *', ' ', text_in)
			text_in = text_in.replace('\n', ' ')
			text_in = text_in.lstrip().rstrip()
			text_in = text_in.lower()


			yield text_in

	def preprocess_in(self, input_text=None):

		self.inputs = input_text

		prep_inputs = []
		for text_in in self.preprocess_in_generator():
			prep_inputs.append(text_in)

		print("prep_inputs[0] : %s"%(prep_inputs[0]))
		sys.stdout.flush()

		print("prep_inputs[1] : %s"%(prep_inputs[1]))
		sys.stdout.flush()

		print("prep_inputs[2] : %s"%(prep_inputs[2]))
		sys.stdout.flush()

		return prep_inputs


	def preprocess_out_generator(self):

		for kps in self.outputs:

			prep_kps = []
			for kp in kps:

				kp = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', kp)
				kp = re.sub(r'/', ' / ', kp) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
				kp = re.sub(r'@\w+', '<USER>', kp)
				kp = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' ', kp) # eliminate numbers
				kp = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', kp) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
				kp = re.sub(r'[^\x00-\x7f]', ' ', kp) # encoded characters
				punct_list = str.maketrans({key: None for key in self.input_punct})
				kp = kp.translate(punct_list)
				kp = re.sub(r'[\-\_\.\?\:\;]+\ *', ' ', kp)
				kp = kp.replace('\n', ' ')
				kp = kp.lstrip().rstrip()
				kp = kp.lower()
				# only for preprocessing output sequences (key phrase)
				# e.g. avoiding to remove 'c++', 'c#'
				regex = re.compile('[%s]' % re.escape(self.kp_punct))
				kp = regex.sub(' ', kp)

				prep_kps.append(kp)

			prep_kps = list(set(prep_kps))

			yield prep_kps # preprocessed / clean version of raw keyphrase list

	def preprocess_out(self, output_keyphrases=None):

		self.outputs = output_keyphrases

		prep_outputs =[]
		for text_out in self.preprocess_out_generator():
			prep_outputs.append(text_out)

		print("prep_outputs[0] : %s"%(prep_outputs[0]))
		sys.stdout.flush()

		print("prep_outputs[1] : %s"%(prep_outputs[1]))
		sys.stdout.flush()

		print("prep_outputs[2] : %s"%(prep_outputs[2]))
		sys.stdout.flush()

		return prep_outputs



	def tokenize_in_generator(self):

		
		for inputs in self.prep_inputs:

			tokens_ = inputs.split()
			tokens = []
			# discard tokens which len is < 2 and >= 20
			for t in tokens_:

				regex = re.compile('[%s]' % re.escape(self.kp_punct))
				t = regex.sub('', t)
				### no stemming
				#t = sno.stem(t)

				if (len(t) > 1) and (len(t) < 20):
					
					tokens.append(t)

			yield tokens

	def tokenize_in(self, prep_inputs=None):

		self.prep_inputs = prep_inputs

		inputs_tokens = []

		for tokens in self.tokenize_in_generator():
			inputs_tokens.append(tokens)

		print("inputs_tokens[0] : %s"%(inputs_tokens[0]))
		sys.stdout.flush()

		return inputs_tokens

	def tokenize_out_generator(self):

		for kps in self.prep_outputs:

			token_kps = []
			for kp in kps:
				kp_tokens = kp.split()
				tokens = []
				# only store N-grams of keyphrase, which N < 7.
				# if >> mostly noises

				if (len(kp_tokens) >= 1) and (len(kp_tokens) < 10):
					# discard tokens which character len is < 2 and >= 20
					for t in kp_tokens:

						regex = re.compile('[%s]' % re.escape(self.kp_punct))
						t = regex.sub('', t)
						### no stemming
						#t = sno.stem(t)
						if(len(t) != 0):
							if len(t) > 1 and len(t) < 20:
								tokens.append(t)

				if(len(tokens) != 0):
					token_kps.append(tokens)

			yield token_kps



	def tokenize_out(self, prep_outputs=None):

		self.prep_outputs = prep_outputs

		output_tokens = []

		for tokens in self.tokenize_out_generator():
			output_tokens.append(tokens)

		print("output_tokens[0] : %s"%(output_tokens[0]))
		sys.stdout.flush()

		return output_tokens


	def all_tokens_generator(self):

		for (in_tokens, out_tokens) in zip(self.inputs_tokens, self.output_tokens):
			tokens = []
			tokens.extend(in_tokens)
			for kps in out_tokens:
				for kp in kps:
					tokens.append(kp)

			yield tokens

	def get_all_tokens(self, inputs_tokens=None, output_tokens=None):

		self.inputs_tokens = inputs_tokens
		self.output_tokens = output_tokens

		all_tokens = []
		for tokens in self.all_tokens_generator():
			all_tokens.extend(tokens)

		return all_tokens


	def split_sent_generator(self):

		for text_in in self.inputs_tokens:

			text_in = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', text_in)
			text_in = re.sub(r'/', ' / ', text_in) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
			text_in = re.sub(r'@\w+', '<USER>', text_in)
			text_in = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' ', text_in) # eliminate numbers
			text_in = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', text_in) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			text_in = re.sub(r'[^\x00-\x7f]', ' ', text_in) # encoded characters
			punct_list = str.maketrans({key: None for key in self.input_punct})
			text_in = text_in.translate(punct_list)
			text_in = text_in.replace('\n', ' ')
			text_in = text_in.lstrip().rstrip()
			text_in = text_in.lower()

			# split into sentences 

			sents = re.split(r'(?<!\w\.\,\w.,)(?<![A-Z][a-z]\.\,)(?<=\.|\,|\?)\s', text_in)


			yield sents

	def split_sent(self, input_text=None):

		self.inputs_tokens = input_text
		doc_sents = []
		for sents in self.split_sent_generator():
			doc_sents.append(sents)

		print("doc_sents[0]: %s"%doc_sents[0])

		return doc_sents


	def tokenized_sent_generator(self):

		for doc in self.inputs_tokens:
			sent_tokens = []
			for sent in doc:
				tokens = sent.split()
				tokens_ = []
				if len(tokens) > 3:
					for t in tokens:
						regex = re.compile('[%s]' % re.escape(self.kp_punct))
						t = regex.sub('', t)

						### no stemming
						#t = sno.stem(t)
						if (len(t) > 1) and (len(t) < 20):
							tokens_.append(t)
				if (len(tokens_) != 0):
					sent_tokens.append(tokens_)
			yield sent_tokens


	def tokenized_sent(self, input_text=None):

		self.inputs_tokens = input_text

		tokenized_docs = []
		for tokenized_sent in self.tokenized_sent_generator():
			tokenized_docs.append(tokenized_sent)

		print("tokenized_docs[0]: %s"%tokenized_docs[0])

		return tokenized_docs











