import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import _pickle as cPickle
import h5py
import random
from datetime import datetime
import time
from utils.data_connector import DataConnector


class PrepareEmbedding():

	def __init__(self):

		
		self.glove_embedding = None
		self.oov_embedding = None

	# pretrained_embedding stored in word2vec format
	def create_nontrainable(self, all_vocabularies, pretrained_embeddings, embedding_dim):

		vocab_size = len(all_vocabularies)
		pretrained_size = len(list(pretrained_embeddings.vocab.keys()))
		embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

		print("number of all vocabularies in current corpus index: %s"%vocab_size)

		print("number of vocabularies in pretrained embeddings: %s"%pretrained_size)

		for i in range(embeddings.shape[0]):
			# check whether we can find the word in pretrained embeddings

			word_found = pretrained_embeddings.vocab.get(all_vocabularies[i])

			if word_found != None:
				embeddings[i,:] = pretrained_embeddings[all_vocabularies[i]]
			else:
				continue

		self.glove_embedding = embeddings

		return embeddings

	def create_trainable(self, out_vocabularies, embedding_dim):

		oov_size = len(out_vocabularies)
		print("creating empty matrix for OOV embedding...")

		embeddings = np.zeros((oov_size, embedding_dim), dtype=np.float32)

		self.oov_embedding = embeddings

		return embeddings


