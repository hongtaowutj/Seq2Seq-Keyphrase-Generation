#!/usr/bin/anaconda3/bin/python
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
from math import log
import json
from collections import OrderedDict
from prep_moduls_wo import *
import nltk
nltk.data.path.append('/home/TUE/inimah/nltk_data')
sno = nltk.stem.SnowballStemmer('english')

DATA = 'data'
eof_punc = ['.','?','!',',',';','-','_','+', '#']
punc = [x for x in string.punctuation if x not in eof_punc]

unlisted_punct = ['-', '_', '+', '#']
punct = ''.join([p for p in string.punctuation if p not in unlisted_punct])


def cleaning_text(text, punc):
	
	text = clean(striphtml(text)).lstrip().rstrip().lower()
	# remove number from text data
	#text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', '', text)
	text = preprocess(text)
	# remove punctuation from 
	punct_list = str.maketrans({key: None for key in punc})
	text = text.translate(punct_list)
	text = text.replace('\n', '')
	text = text.lstrip().rstrip()
	text = text.lower()
	text = sno.stem(text)
	
	return text    


if __name__ == '__main__':

	training_data = []
	for line in open(os.path.join(DATA,'kp20k_training.json'), 'r'):
		training_data.append(json.loads(line))

	'''

	validation_data = []
	for line in open(os.path.join(DATA,'kp20k_validation.json'), 'r'):
		validation_data.append(json.loads(line))

	testing_data = []
	for line in open(os.path.join(DATA,'kp20k_testing.json'), 'r'):
		testing_data.append(json.loads(line))
	'''

	all_words_train = []

	tokenized_train_data = OrderedDict()

	for i, train_data in enumerate(training_data):

		
		clean_title = cleaning_text(train_data['title'], punc)
		clean_abstract = cleaning_text(train_data['abstract'], punc)
		clean_keywords = cleaning_text(train_data['keyword'], punc)
		
		all_words_train.extend(tokenizeWords(clean_title))
		all_words_train.extend(tokenizeWords(clean_abstract))
		all_words_train.extend(tokenizeWords(clean_keywords))
		
		title_tokens = tokenizeWords(clean_title)
		abstract_tokens = tokenizeWords(clean_abstract)
		keys = clean_keywords.split(';')
		keywords_tokens = []
		for kp in keys:
			keywords_tokens.append(tokenizeWords(kp))

		tokenized_train_data[i] = [title_tokens, abstract_tokens, keywords_tokens]

	'''

	tokenized_valid_data = OrderedDict()

	for i, valid_data in enumerate(validation_data):

		clean_title = cleaning_text(valid_data['title'], punc)
		clean_abstract = cleaning_text(valid_data['abstract'], punc)
		clean_keywords = cleaning_text(valid_data['keyword'], punc)
		
		all_words.extend(tokenizeWords(clean_title))
		all_words.extend(tokenizeWords(clean_abstract))
		all_words.extend(tokenizeWords(clean_keywords))
		
		title_tokens = tokenizeWords(clean_title)
		abstract_tokens = tokenizeWords(clean_abstract)
		keys = clean_keywords.split(';')
		keywords_tokens = []
		for kp in keys:
			keywords_tokens.append(tokenizeWords(kp))

		tokenized_valid_data[i] = [title_tokens, abstract_tokens, keywords_tokens]

	tokenized_test_data = OrderedDict()

	for i, test_data in enumerate(testing_data[:100]):

		clean_title = cleaning_text(test_data['title'], punc)
		clean_abstract = cleaning_text(test_data['abstract'], punc)
		clean_keywords = cleaning_text(test_data['keyword'], punc)
		
		all_words.extend(tokenizeWords(clean_title))
		all_words.extend(tokenizeWords(clean_abstract))
		all_words.extend(tokenizeWords(clean_keywords))
		
		title_tokens = tokenizeWords(clean_title)
		abstract_tokens = tokenizeWords(clean_abstract)
		keys = clean_keywords.split(';')
		keywords_tokens = []
		for kp in keys:
			keywords_tokens.append(tokenizeWords(kp))

		tokenized_test_data[i] = [title_tokens, abstract_tokens, keywords_tokens]
	'''

	savePickle(all_words_train, os.path.join(DATA,'all_words_train_wo'))
	savePickle(tokenized_train_data, os.path.join(DATA,'tokenized_train_data_wo'))
	#savePickle(tokenized_valid_data, os.path.join(DATA,'tokenized_valid_data'))
	#savePickle(tokenized_test_data, os.path.join(DATA,'tokenized_test_data'))


