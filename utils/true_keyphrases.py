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

class TrueKeyphrases():

	def __init__(self, true_keyphrase):

        self.y_pred = y_pred
        self.true_keyphrase = true_keyphrase # tokenized y_true (need to be joined)
        self.filepath = filepath
        
        self.y_true = []
        self.max_kp_num = []
        self.mean_kp_num = []
        self.std_kp_num = []


    '''
    y_true parameter is tokenized version of ground truth keyphrase list
    need to transform it back to its original keyphrase form

    '''
    def get_true_keyphrases(self):

        all_kps = []
        # for each set of keyphrases given text inputs
        for kp_list in self.true_keyphrase:
            kps = []
            for tokenized_kp in kp_list:
                kp = " ".join(tokenized_kp)
                kps.append(kp)
            all_kps.append(kps)

        self.y_true = all_kps

        return self.y_true # y_true is list of set keyphrases in corpus

    def get_stat_keyphrases(self):

        len_kps = []
        for kps_list in self.y_true:
            len_kps.append(len(kps_list))

        self.max_kp_num = max(len_kps)
        self.mean_kp_num = np.mean(np.array(len_kps))
        self.std_kp_num = np.std(np.array(len_kps))

        return self.max_kp_num, self.mean_kp_num, self.std_kp_num