# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from collections import defaultdict
import string
from string import punctuation
import re
import nltk
import _pickle as cPickle

unlisted_punct = ['-', '_', '+', '#']
punct = ''.join([p for p in string.punctuation if p not in unlisted_punct])

def clean_keyphrases(keyphrase_list):
  
  kp_list = []
  
  for kp in keyphrase_list:
    
    regex = re.compile('[%s]' % re.escape(punct))
    text = regex.sub('', kp)
    text = re.sub(r"[\-\_]+\ *", " ", text)
    text = text.lower()
    
    kp_list.append(text)

  return kp_list  



"""
Function to map id in validation/test set to the 
ground truth list of key phrases in original preprocessed data set
"""

def map_ids(doc_ids, train_docids, rand_ids):
  
  ori_docids = []
  
  for idx in doc_ids:
    # get original ids of validation/test set, since it is randomly shuffled 
    set_id = rand_ids[idx]
    ori_docids.append(train_docids[set_id])
    
  return ori_docids   



"""
Get ground truth list of key phrases
"""

def get_keyphrases(doc_ids, doc_topics):
  
  all_keyphrases = []
  
  for idx in doc_ids:
    
    keyps = []
    kp = doc_topics[idx][2]
    
    for (idkey, keyp) in kp:
      keyps.append(keyp)
      
    # clean from punctuation
    keyps = clean_keyphrases(keyps)
    keyps = [kp.rstrip() for kp in keyps]
      
    all_keyphrases.append(keyps)
        
  return all_keyphrases 



def get_keyphrases_alldocs(doc_ids, doc_topics):
  
  all_keyphrases = []
  
  for idx in doc_ids:
    
    keyps = []
    kp = doc_topics[idx][2]
    
    for keyp in kp:
      keyps.append(keyp)
      
    # clean from punctuation
    keyps = clean_keyphrases(keyps)
    keyps = [kp.lstrip().rstrip() for kp in keyps]
      
    all_keyphrases.append(keyps)
        
  return all_keyphrases

def get_keyphrases_kp20k(doc_topics):
  
  all_keyphrases = []
  
  for k, v in doc_topics.items():
    keyps = []
    kp = v[2]
   
    for keyp in kp:
      keyp = " ".join(keyp)
      keyps.append(keyp)

    all_keyphrases.append(keyps)
        
  return all_keyphrases 

"""
to check whether model generated duplicates
"""

def count_duplicates(keyphrase_list):
  
  tally = defaultdict(int)
  for kp in keyphrase_list: 
    tally[kp] += 1
    
  num_duplicates = sum(list(tally.values())) - len(list(tally.values()))
  
  return num_duplicates


def get_stat_keyphrases(keyphrases_list):
  
  len_kps = []
  for kps in keyphrases_list:
    len_kps.append(len(kps))
  
  max_kps = max(len_kps)
  mean_kps = np.mean(np.array(len_kps))
  std_kps = np.std(np.array(len_kps))
  
  return max_kps, mean_kps, std_kps


"""
Evaluation metrics:

1. True Positive

"""

def get_TP(kp_true, kp_pred):
  
  list_tp = list(set(kp_pred) & set(kp_true)) 
  tp = len(list_tp)
  
  return tp, list_tp


"""
2. False Negative

"""

def get_FN(kp_true, kp_pred):
  
  list_fn = list(set(kp_true) - set(kp_pred))
  fn = len(list_fn)
  
  return fn, list_fn  


"""
3. False Positive
"""

def get_FP(kp_true, kp_pred):
  
  list_fp = list(set(kp_pred) - set(kp_true))
  fp = len(list_fp)
  
  return fp, list_fp


"""
4. Accuracy

"""

def get_accuracy(tp, fp, fn):
  
  accuracy = tp / (tp + fn + fp)
  
  return accuracy


"""
5. Precision

"""

def get_precision(tp, fp, fn):
  
  precision = tp / (tp + fp)
  
  return precision

"""
5. Recall

"""

def get_recall(tp, fp, fn):
  
  recall = tp / (tp + fn)
  
  return recall

"""
6. F-measure
beta = 1 --> F1-score (harmonic mean of precision and recall)

"""


def get_fscore(beta, tp, fp, fn):
  
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  
  fscore = (beta**2 + 1) * precision * recall / (beta * precision + recall)
  
  return fscore


def get_mean_evals(all_acc, all_precision, all_recall, all_fscore):
  
  avg_acc = np.mean(np.array(all_acc))
  avg_precision = np.mean(np.array(all_precision))
  avg_recall = np.mean(np.array(all_recall))
  avg_fscore = np.mean(np.array(all_fscore))
  
  return avg_acc, avg_precision, avg_recall, avg_fscore

def get_mean_cm(all_tps, all_fps, all_fns):
  
  avg_tps = np.mean(np.array(all_tps))
  avg_fps = np.mean(np.array(all_fps))
  avg_fns = np.mean(np.array(all_fns))

  
  return avg_tps, avg_fps, avg_fns
