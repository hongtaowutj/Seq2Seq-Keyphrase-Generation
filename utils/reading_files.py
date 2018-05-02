# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
sys.path.append(os.getcwd())
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


class ReadingFiles():

	def __init__(self, filepath, filename_to_save):

		self.filepath = filepath
		self.filename_to_save = filename_to_save
		self.filenames = []
		self.doc_xml = OrderedDict()
		self.doc_txt = OrderedDict()
		self.author_keys = OrderedDict()
		self.reader_keys = OrderedDict()
		self.keyphrases = OrderedDict()
		self.all_articles = OrderedDict()
		self.all_doc_keyphrases = OrderedDict()

		kp_unlisted_punct = ['-', '_', '+', '#']
		self.kp_punct = ''.join([p for p in string.punctuation if p not in kp_unlisted_punct])
		eof_punc = ['.','?','!',',',';','-','_','+', '#']
		self.input_punct = [x for x in string.punctuation if x not in eof_punc]


	def listing_files(self):
		filenames = []
		for path, subdirs, files in os.walk(self.filepath):
			for name in files:
				filenames.append(os.path.join(path, name))

		self.filenames = filenames


	'''
	For DATA: Nguyen2007 (NUS)
	'''
	def construct_dirinfos(self):

		doc_xml = OrderedDict()
		doc_txt = OrderedDict()
		author_keys = OrderedDict()
		reader_keys = OrderedDict()

		def constructing(filename):

			head_dir, tail_dir = os.path.split(os.path.split(filename)[0])

			if tail_dir != 'KEY':
				docid = tail_dir
			else:
				docid = os.path.split(head_dir)[1]
				
			doc_xml[docid] = []
			doc_txt[docid] = []
			author_keys[docid] = []
			reader_keys[docid] = []

			return 0

		def assigning_fileinfos(filename):

			head_dir, tail_dir = os.path.split(os.path.split(filename)[0])
			if tail_dir != 'KEY':
				docid = tail_dir
			else:
				docid = os.path.split(head_dir)[1]
			
			nameoffile = os.path.basename(filename)
			fileName, fileExtension = os.path.splitext(nameoffile)
			
			if fileExtension == '.txt':
				doc_txt[docid].append(filename)
			elif fileExtension == '.xml':
				doc_xml[docid].append(filename)
			elif fileExtension == '.kwd':
				author_keys[docid].append(filename)
			elif fileExtension == '.key':
				reader_keys[docid].append(filename)

			return 0

		for i, nof in enumerate(self.filenames):
			constructing(nof)

		for i, nof in enumerate(self.filenames):
			assigning_fileinfos(nof)

		self.doc_xml = doc_xml
		self.doc_txt = doc_txt
		self.author_keys= author_keys
		self.reader_keys = reader_keys

	
	'''
	For DATA: Nguyen2007 (NUS)
	'''
	def merging_keys(self):

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

		def read_keyphrases(filepath):

			keyphrases = set()
			for path in filepath:
				with open(path, encoding="utf8", errors='ignore') as f:
					for i, line in enumerate(f):
						kp = cleaning(line)
						if kp not in keyphrases:
							keyphrases.add(kp)          
			return keyphrases    

		a_keyphrases = OrderedDict()
		for k, v in self.author_keys.items():
			keyphrases = list(read_keyphrases(v))
			a_keyphrases[int(k)] = keyphrases

		r_keyphrases = OrderedDict()
		for k, v in self.reader_keys.items():
			keyphrases = list(read_keyphrases(v))
			r_keyphrases[int(k)] = keyphrases

		ar_keyphrases = OrderedDict()
		for k,v in a_keyphrases.items():
			a_kps = v
			r_kps = r_keyphrases[k]
			merged = list(set(np.append(a_kps,r_kps)))
			ar_keyphrases[k] = merged

		self.keyphrases = ar_keyphrases


	'''
	For DATA: Nguyen2007 (NUS)
	'''

	def reading_texts(self):

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

		def reading(filename):

			textTitle = ""
			textAbstract = ""
			textMain = ""
			line_abstract = 0
			line_intro = 0
			line_ref = 0
			line_ack = 0

			with open(filename, encoding="utf8", errors='ignore') as f:

				for i,line in enumerate(f):
					if line.strip().lower() == "abstract":
						line_abstract = i
					if "introduction" in list(line.strip().lower().split()):
						line_intro = i
					if "references" in list(line.strip().lower().split()):
						line_ref = i
					if "acknowledgements" in list(line.strip().lower().split()):
						line_ack = i 

			# retrieving text from the following sections of an article:
			# - title
			# - abstract
			# - main content (introduction, etc.. beside acknowledgement and references)

			with open(filename, encoding="utf8", errors='ignore') as f:
				
				for i,line in enumerate(f):
					# retrieve title
					if i == 0:
						txt = cleaning(line)
						if(len(txt) != 0):
							textTitle = txt

					# retrieve text from abstract
					if (i > line_abstract) and (i < line_intro):
						txt = cleaning(line)
						if(len(txt) != 0):
							textAbstract +=  txt + " "
					# retrieve text from introduction section to acknowledgement (if this section exists)
					if (line_ack != 0):
						if (i > line_intro) and (i < line_ack):   
							txt = cleaning(line)
							if(len(txt) != 0):
								textMain +=  txt + " "
					# retrieve text from introduction section to references
					else:
						if (i > line_intro) and (i < line_ref):   
							txt = cleaning(line) 
							if(len(txt) != 0):
								textMain +=   txt + " "


			return textTitle, textAbstract, textMain

		all_docs = OrderedDict()
		for k,v in self.doc_txt.items():
			docid = int(k)
			path = v[0]
			textTitle, textAbstract, textMain = reading(path)
			all_docs[docid] = [textTitle, textAbstract, textMain]

		self.all_articles = all_docs

	'''
	For DATA: Krapivin
	'''

	def reading_krapivin(self):

		doc_txt = OrderedDict()
		keyphrases = OrderedDict()

		def get_file_infos(filepath):
	
			nameoffile = os.path.basename(filepath)
			fileName, fileExtension = os.path.splitext(nameoffile)
		
			if fileExtension == '.txt':
				doc_txt[int(fileName)] = filepath
			elif fileExtension == '.key':
				keyphrases[int(fileName)] = filepath

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

		def reading(filename):

			textTitle = ""
			textAbstract = ""
			textMain = ""
			line_title = 0
			line_abstract = 0
			line_intro = 0
			line_ref = 0
			line_ack = 0

			with open(filename, encoding="utf8", errors='ignore') as f:

				for i,line in enumerate(f):
			
					if line.strip() == "--T":
						line_title = i+1            
					elif line.strip() == "--A":
						line_abstract = i+1
					elif line.strip() == "--B":
						line_intro = i+2
					elif line.strip() == "--R":
						line_ref = i
					elif line.strip().lower() == "acknowledgements":
						line_ack = i 


			# retrieving text from the following sections of an article:
			# - title
			# - abstract
			# - main content (introduction, etc.. beside acknowledgement and references)

			with open(filename, encoding="utf8", errors='ignore') as f:
				
				for i,line in enumerate(f):
					
					# retrieve title
					if i == line_title:
						txt = cleaning(line)
						if(len(txt) != 0):
							textTitle = txt

					# retrieve text from abstract
					if (i >= line_abstract) and (i < line_intro-2):
						txt = cleaning(line)
						if(len(txt) != 0):
							textAbstract +=  txt + " "
					# retrieve text from introduction section to acknowledgement (if this section exists)
					if (line_ack != 0):
						if (i >= line_intro) and (i < line_ack):   
							txt = cleaning(line)
							if(len(txt) != 0):
								textMain +=  txt + " "
					# retrieve text from introduction section to references
					else:
						if (i >= line_intro) and (i < line_ref):   
							txt = cleaning(line) 
							if(len(txt) != 0):
								textMain +=   txt + " "


			return textTitle, textAbstract, textMain

		def read_keyphrases(filepath):

			keyphrases = set()
			for path in filepath:
				with open(path, encoding="utf8", errors='ignore') as f:
					for i, line in enumerate(f):
						kp = cleaning(line)
						if kp not in keyphrases:
							keyphrases.add(kp)          
			return keyphrases


		for i, nof in enumerate(self.filenames):
			get_file_infos(nof)

		all_docs = OrderedDict()
	
		for k,v in doc_txt.items():
			docid = int(k)
			path = v[0]
			textTitle, textAbstract, textMain = reading(path)
			all_docs[docid] = [textTitle, textAbstract, textMain]


		self.all_articles = all_docs

		all_keyphrases = OrderedDict()

		for k,v in keyphrases.items():
			docid = int(k)
			path = v[0]
			kp_set = read_keyphrases(path)
			all_keyphrases[docid] = [kp_set]

		self.keyphrases = all_keyphrases



	'''
	merging text sources and keyphrase topics 
	'''
	def merging_data(self):

		doc_topics = OrderedDict()
		for k,v in self.all_articles.items():
			title = v[0]
			abstract = v[1]
			maintext = v[2]
			topics = self.keyphrases[k]
			doc_topics[k] = [title, abstract, maintext, topics]

		self.all_doc_keyphrases = doc_topics


	def save_files(self):

		def saving_pickles(data, filename):

			f = open(filename, 'wb')
			cPickle.dump(data, f)
			f.close()

			print(" file saved to: %s"%filename)

		# save indices_words
		saving_pickles(self.all_doc_keyphrases, os.path.join(self.filepath, self.filename_to_save))



	


	