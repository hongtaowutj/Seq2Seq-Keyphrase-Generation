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
    def __init__(self, word_tokens, filepath):

        self.word_tokens = word_tokens
        self.filepath = filepath
        self.indices_words = []
        self.words_indices = []
        self.term_freq = []
       
    def vocabulary_indexing(self):

        self.term_freq = nltk.FreqDist(self.word_tokens)
        wordIndex = list(self.term_freq.keys())
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