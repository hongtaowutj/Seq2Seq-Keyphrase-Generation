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
	def __init__(self, input_text, output_keyphrases, filepath, filename):

		self.inputs = input_text
		self.outputs = output_keyphrases
		self.filepath = filepath
		self.filename = filename
		kp_unlisted_punct = ['-', '_', '+', '#']
		self.kp_punct = ''.join([p for p in string.punctuation if p not in kp_unlisted_punct])
		eof_punc = ['.','?','!',',',';','-','_','+', '#']
		self.input_punct = [x for x in string.punctuation if x not in eof_punc]
		self.prep_inputs = []
		self.prep_outputs = []
		self.inputs_tokens = [] 
		self.output_tokens = []

	def preprocess_articles(self):

		def cleaning(text):

			text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', text)
			text = re.sub(r'/', ' / ', text) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
			text = re.sub(r'@\w+', '<USER>', text)
			text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', "", text) # eliminate numbers
			text = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', text) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			text = re.sub(r'[^\x00-\x7f]', '', text) # encoded characters
			punct_list = str.maketrans({key: None for key in self.input_punct})
			text = text.translate(punct_list)
			text = re.sub(r'[\-\_\.\?]+\ *', ' ', text)
			text = text.replace('\n', '')
			text = text.lstrip().rstrip()
			text = text.lower()
			
			return text

		text_in = self.inputs
		self.prep_inputs = cleaning(text_in) # preprocessed / clean version of raw text
		kps = self.outputs
		prep_kps = []
		for kp in kps:
			kp = cleaning(kp)
			regex = re.compile('[%s]' % re.escape(self.kp_punct))
			kp = regex.sub('', kp)
			prep_kps.append(kp)
		self.prep_outputs = prep_kps # preprocessed / clean version of raw keyphrase list

		return self.prep_inputs, self.prep_outputs


	def preprocess_tweets(self):

		def cleaning(text):

			def split_hashtag(found):
				hashtag_body = found.group(0)[1:]
			return "<HASHTAG> " + hashtag_body + " <ALLCAPS>"

			# Different regex parts for smiley faces
			eyes = "[8:=;]"
			nose = "['`\-]?"

			text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '<URL>', text)
			text = re.sub(r'/', ' / ', text) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
			text = re.sub(r'@\w+', '<USER>', text)
			text = re.sub(eyes + nose + r'[)dD]+|[(dD]+' + nose + eyes, "<SMILE>", text)
			text = re.sub(eyes + nose + r'[pP]+', "<LOLFACE>", text)
			text = re.sub(eyes + nose + r'\(+|\)+' + nose + eyes, "<SADFACE>", text)
			text = re.sub(eyes + nose + r'( \/|[\\|l*])', "<NEUTRALFACE>", text)
			text = re.sub(r'<3', "<HEART>", text)
			text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', "<NUMBER>", text)
			text = re.sub(r'#\S+', split_hashtag, text) # Split hashtags on uppercase letters
			text = re.sub(r'([!?.]){2,}', r'\1 <REPEAT>', text) # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
			text = re.sub(r'\b(\S*?)(.)\2{2,}\b', r'\1\2 <ELONG>', text) # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
			text = re.sub(r'[^\x00-\x7f]', '', text) # encoded characters
			punct_list = str.maketrans({key: None for key in self.input_punct})
			text = text.translate(punct_list)
			text = re.sub(r'[\-\_\.\?]+\ *', ' ', text)
			text = text.replace('\n', '')
			text = text.lstrip().rstrip()
			text = text.lower()
		 
			return text

		text_in = self.inputs
		self.prep_inputs = cleaning(text_in) # preprocessed / clean version of raw text
		kps = self.outputs
		prep_kps = []
		for kp in kps:
			kp = cleaning(kp)
			regex = re.compile('[%s]' % re.escape(self.kp_punct))
			kp = regex.sub('', kp)
			prep_kps.append(kp)
		self.prep_outputs = prep_kps # preprocessed / clean version of raw keyphrase list

		return self.prep_inputs, self.prep_outputs

	def tokenize_words(self):

		def tokenization(text):

			tokens_ = text.split()
			tokens = []
			# discard tokens which len is < 2 and >= 20
			for t in tokens_:
				if len(t) > 1 and len(t) < 20:
					t = sno.stem(t)
					tokens.append(t)

			return tokens

		prep_inputs = self.prep_inputs
		prep_outputs = self.prep_outputs # list of keyphrase
		self.inputs_tokens = tokenization(prep_inputs) # tokenized input text (model input)
		kps = []
		for kp in prep_outputs:
			kp_tokens = tokenization(kp)
			kps.append(kp_tokens)
		self.output_tokens = kps # tokenized keyphrase list (model output)

		return self.inputs_tokens, self.output_tokens

