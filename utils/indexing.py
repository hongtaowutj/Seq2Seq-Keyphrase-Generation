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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class Indexing():

    # input here is individual text
    def __init__(self):

        self.word_tokens = []
        self.filepath = ""
        self.filename = ""
        self.indices_words = {}
        self.words_indices = {}
        self.term_freq = {}
       
    def vocabulary_indexing(self, word_tokens):

        self.word_tokens = word_tokens

        term_freq = nltk.FreqDist(word_tokens)
        self.term_freq = term_freq

        wordIndex = list(term_freq.keys())
        wordIndex.insert(0,'<pad>')
        wordIndex.append('<start>')
        wordIndex.append('<end>')
        wordIndex.append('<unk>')

        # indexing word vocabulary : pairs of (index,word)
        vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])
        self.indices_words = vocab
        self.words_indices = dict((v,k) for (k,v) in self.indices_words.items())

        return self.term_freq, self.indices_words, self.words_indices

    def zipf_plot(self):

        counts = list(self.term_freq.values())
        tokens = list(self.term_freq.keys())

        ranks = np.arange(1, len(counts)+1)
        indices = np.argsort(-counts)
        frequencies = counts[indices]

        plt.clf()
        plt.loglog(ranks, frequencies, marker=".")
        plt.title("Zipf plot")
        plt.xlabel("Frequency rank of token")
        plt.ylabel("Absolute frequency of token")
        plt.grid(True)
        for n in list(np.logspace(-0.5, np.log10(len(counts)), 20).astype(int)):
            dummy = plt.text(ranks[n], frequencies[n], " " + tokens[indices[n]], 
                         verticalalignment="bottom",
                         horizontalalignment="left")

        plt.savefig(os.path.join(self.filepath,'zipf_plot.png'))



    def save_files(self):

        
        def saving_pickles(data, filename):

            f = open(filename, 'wb')
            cPickle.dump(data, f)
            f.close()

            print(" file saved to: %s"%filename)

        # save indices_words
        saving_pickles(self.indices_words, os.path.join(self.filepath,'indices_words.pkl'))
        saving_pickles(self.words_indices, os.path.join(self.filepath,'words_indices.pkl'))
        saving_pickles(self.term_freq, os.path.join(self.filepath,'term_freq.pkl'))

    def save_(self, data, filename, filepath):


        f = open(os.path.join(filepath, filename), 'wb')
        cPickle.dump(data, f)
        f.close()

        print(" file saved to: %s"%(os.path.join(filepath, filename)))

   
        

    def sampling_table(self, size, sampling_factor=1e-5):

        """

        Generates a word rank-based probabilistic sampling table.
        Used for generating the `sampling_table` argument for `skipgrams`.
        `sampling_table[i]` is the probability of sampling
        the word i-th most common word in a dataset
        (more common words should be sampled less frequently, for balance).
        The sampling probabilities are generated according
        to the sampling distribution used in word2vec:
        `p(word) = min(1, sqrt(word_frequency / sampling_factor) / (word_frequency / sampling_factor))`
        We assume that the word frequencies follow Zipf's law (s=1) to derive
        a numerical approximation of frequency(rank):
        `frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`
        where `gamma` is the Euler-Mascheroni constant.

        # Arguments
            size: Int, number of possible words to sample.
            sampling_factor: The sampling factor in the word2vec formula.
        # Returns
            A 1D Numpy array of length `size` where the ith entry
            is the probability that a word of rank i should be sampled.
        """

        gamma = 0.577
        rank = np.arange(1, size)
        inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
        f = sampling_factor * inv_fq

        return np.minimum(1., f / np.sqrt(f))
