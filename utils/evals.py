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


class Evaluate():

    '''
    input is list of all generated keyphrases from corpus
    and true values of keyphrases
    '''
    def __init__(self, y_pred, true_keyphrase, filepath):

        self.y_pred = y_pred
        self.true_keyphrase = true_keyphrase # tokenized y_true (need to be joined)
        self.filepath = filepath
        
        self.y_true = []
        self.max_kp_num = []
        self.mean_kp_num = []
        self.std_kp_num = []

        self.tp = []
        self.tp_list = []
        self.fn = []
        self.fn_list = []
        self.fp = []
        self.fp_list = []

        self.mean_tps = [] 
        self.mean_fns = [] 
        self.mean_fps = [] 

        self.accuracy= []
        self.precision = []
        self.recall = []
        self.fscore = []

        self.mean_acc = [] 
        self.mean_precision = [] 
        self.mean_recall = [] 
        self.mean_fscore = []


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

    def compute_true_positive(self):

        for i in range(len(self.y_pred)):
            tp_list = list(set(self.y_pred[i]) & set(self.y_true[i]))
            self.tp_list.append(tp_list)
            self.tp.append(len(tp_list))

        return self.tp, self.tp_list # all tps: tp is computed per document in corpus

    def compute_false_negative(self):

        for i in range(len(self.y_pred)):
            fn_list = list(set(self.y_true[i]) - set(self.y_pred[i]))
            self.fn_list.append(fn_list)
            self.fn.append(len(fn_list))

        return self.fn, self.fn_list # all fns: tp is computed per document in corpus

    def compute_false_positive(self):

        for i in range(len(self.y_pred)):
            fp_list = list(set(self.y_pred[i]) - set(self.y_true[i]))
            self.fp_list.append(fp_list)
            self.fp.append(len(fp_list))

        return self.fp, self.fp_list # all fps: tp is computed per document in corpus

    def compute_accuracy(self):

        for i in range(len(self.tp)):
            accuracy = self.tp[i] / (self.tp[i] + self.fn[i] + self.fp[i])
            self.accuracy.append(accuracy)

        return self.accuracy

    def compute_precision(self):

        for i in range(len(self.tp)):        
             precision = self.tp[i] / (self.tp[i] + self.fp[i])
             self.precision.append(precision)

        return self.precision

    def compute_recall(self):

        for i in range(len(self.tp)):
             recall = self.tp[i] / (self.tp[i] + self.fn[i])
             self.recall.append(recall)

        return self.recall

    def compute_fscore(self, beta=1):

        for i in range(len(self.precision)):
             fscore = (beta**2 + 1) * self.precision[i] * self.recall[i] / (beta * self.precision[i] + self.recall[i])
             self.fscore.append(fscore)

        return self.fscore

    def compute_mean_evals(self):

        self.mean_acc = np.mean(np.array(self.accuracy))
        self.mean_precision = np.mean(np.array(self.precision))
        self.mean_recall = np.mean(np.array(self.recall))
        self.mean_fscore = np.mean(np.array(self.fscore))

        return self.mean_acc, self.mean_precision, self.mean_recall, self.mean_fscore

    def compute_mean_cm(self):

        self.mean_tps = np.mean(np.array(self.tp))
        self.mean_fns = np.mean(np.array(self.fn))
        self.mean_fps  = np.mean(np.array(self.fp))

        return self.mean_tps, self.mean_fns, self.mean_fps  

    def print_mean_evals(self):

        print("Average Accuracy: %s"%self.mean_acc)
        print("Average Precision: %s"%self.mean_precision)
        print("Average Recall: %s"%self.mean_recall)
        print("Average F1-score: %s"%self.mean_fscore)

    def save_files(self):

        def saving_pickles(data, filename):

            f = open(filename, 'wb')
            cPickle.dump(data, f)
            f.close()

            print(" file saved to: %s"%filename)

        saving_pickles(self.y_true, os.path.join(self.filepath,'y_true_keyphrases.pkl'))

        saving_pickles(self.tp, os.path.join(self.filepath,'all_tps.pkl'))
        saving_pickles(self.fp, os.path.join(self.filepath,'all_fps.pkl'))
        saving_pickles(self.fn, os.path.join(self.filepath,'all_fns.pkl'))

        saving_pickles(self.tp_list, os.path.join(self.filepath,'all_tps_list.pkl'))
        saving_pickles(self.fp_list, os.path.join(self.filepath,'all_fps_list.pkl'))
        saving_pickles(self.fn_list, os.path.join(self.filepath,'all_fns_list.pkl'))

  


    