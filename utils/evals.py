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
        self.max_kp_num = 0
        self.mean_kp_num = 0
        self.std_kp_num = 0

        self.tp = 0
        self.tp_list = []
        self.fn = 0
        self.fn_list = []
        self.fp = 0
        self.fp_list = []

        self.mean_tps = 0 
        self.mean_fns = 0 
        self.mean_fps = 0 

        self.accuracy= 0
        self.precision = 0
        self.recall = 0
        self.fscore = 0

        self.mean_acc = 0 
        self.mean_precision = 0 
        self.mean_recall = 0 
        self.mean_fscore = 0


    '''
    y_true parameter is tokenized version of ground truth keyphrase list
    need to transform it back to its original keyphrase form

    '''

    # return each document true labels of keyphrases into list of unigrams
    def get_true_label_list(self):

        all_kps = []
        # for each set of keyphrases given text inputs
        for kp_list in self.true_keyphrase:
            kps_list = []
            for tokenized_kps in kp_list:
                kps = []
                for kp in tokenized_kps:
                    kps.append(sno.stem(kp))
                kps_list.extend(kps)

            set_kps_list = list(set(kps_list))
            all_kps.append(set_kps_list)

        self.y_true = all_kps

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

        #return self.y_true # y_true is list of set keyphrases in corpus

    def get_true_keyphrases_all(self):

        all_kps = []
        # for each set of keyphrases given text inputs
        for kp_list in self.true_keyphrase:
            kps = []
            for tokenized_kp in kp_list:
                kp = " ".join(tokenized_kp)
                kps.append(kp)
            all_kps.extend(kps)

        self.y_true = all_kps


    def get_stat_keyphrases(self):

        len_kps = []
        for kps_list in self.y_true:
            len_kps.append(len(kps_list))

        self.max_kp_num = max(len_kps)
        self.mean_kp_num = np.mean(np.array(len_kps))
        self.std_kp_num = np.std(np.array(len_kps))

        #return self.max_kp_num, self.mean_kp_num, self.std_kp_num

    def compute_true_positive(self):

        all_tps = []
        all_tps_list = []
        for i in range(len(self.y_pred)):
            tp_list = list(set(self.y_pred[i]) & set(self.y_true[i]))
            all_tps_list.append(tp_list)
            all_tps.append(len(tp_list)+1)

        self.tp_list = all_tps_list
        self.tp = all_tps

        return self.tp, self.tp_list # all tps: tp is computed per document in corpus

    def compute_true_positive_all(self):

        
        tp_list = list(set(self.y_pred) & set(self.y_true))
        all_tps_list = tp_list
        all_tps = len(tp_list) + 1

        print("TP: %s"%all_tps)

        self.tp_list = all_tps_list
        self.tp = all_tps

        return self.tp, self.tp_list

    def compute_false_negative(self):

        all_fns = []
        all_fns_list = []
        for i in range(len(self.y_pred)):
            fn_list = list(set(self.y_true[i]) - set(self.y_pred[i]))
            all_fns_list.append(fn_list)
            all_fns.append(len(fn_list)+1)

        self.fn_list = all_fns_list
        self.fn = all_fns

        return self.fn, self.fn_list # all fns: tp is computed per document in corpus


    def compute_false_negative_all(self):

        
        fn_list = list(set(self.y_true) - set(self.y_pred))
        all_fns_list = fn_list
        all_fns = len(fn_list) + 1

        print("FN: %s"%all_fns)

        self.fn_list = all_fns_list
        self.fn = all_fns

        return self.fn, self.fn_list # all fns: tp is computed from all document in corpus

    def compute_false_positive(self):

        all_fps = []
        all_fps_list = []
        for i in range(len(self.y_pred)):
            fp_list = list(set(self.y_pred[i]) - set(self.y_true[i]))
            all_fps_list.append(fp_list)
            all_fps.append(len(fp_list)+1)

        self.fp_list = all_fps_list
        self.fp = all_fps

        return self.fp, self.fp_list # all fps: tp is computed per document in corpus

    def compute_false_positive_all(self):


        fp_list = list(set(self.y_pred) - set(self.y_true))
        all_fps_list = fp_list
        all_fps = len(fp_list) + 1

        print("FP: %s"%all_fps)

        self.fp_list = all_fps_list
        self.fp = all_fps

        return self.fp, self.fp_list

    def compute_accuracy(self):

        all_acc = []
        for i in range(len(self.tp)):
            accuracy = float(self.tp[i] / (self.tp[i] + self.fn[i] + self.fp[i]))
            all_acc.append(accuracy)

        self.accuracy = all_acc

        return self.accuracy

    def compute_accuracy_all(self):

        accuracy = float(self.tp / (self.tp + self.fn + self.fp)) 

        print("Accuracy: %s"%accuracy)

        self.accuracy = accuracy
        return self.accuracy

    def compute_precision(self):

        all_precision = []
        for i in range(len(self.tp)):        
             precision = float(self.tp[i] / (self.tp[i] + self.fp[i])) 
             all_precision.append(precision)

        self.precision = all_precision
        return self.precision


    def compute_precision_all(self):

        precision = float(self.tp / (self.tp + self.fp)) 

        print("Precision: %s"%precision)

        self.precision = precision
        return self.precision


    def compute_recall(self):

        all_recall = []
        for i in range(len(self.tp)):
             recall = float(self.tp[i] / (self.tp[i] + self.fn[i]))
             all_recall.append(recall)

        self.recall = all_recall
        return self.recall

    def compute_recall_all(self):

        recall = float(self.tp / (self.tp + self.fn)) 

        print("Recall: %s"%recall)

        self.recall = recall
        return self.recall

    def compute_fscore(self, beta=1):

        all_fscore = []
        for i in range(len(self.precision)):
             #fscore = float((beta**2 + 1) * self.precision[i] * self.recall[i] / (beta * self.precision[i] + self.recall[i]))
             fscore = float((2 * self.precision[i] * self.recall[i]) / (self.precision[i] + self.recall[i]))
             #print("F1-score: %s"%fscore)
             #sys.stdout.flush()
             all_fscore.append(fscore)

        self.fscore = all_fscore
        return self.fscore

    def compute_fscore_all(self, beta=1):

        #fscore = float((beta**2 + 1) * self.precision * self.recall / (beta * self.precision + self.recall))

        fscore = float((2 * self.precision * self.recall) / (self.precision + self.recall))

        print("F1-score: %s"%fscore)

        self.fscore = fscore

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

    def print_mean_cm(self):

        print("Average TP: %s"%self.mean_tps)
        print("Average FP: %s"%self.mean_fps)
        print("Average FN: %s"%self.mean_fns)

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

  


    