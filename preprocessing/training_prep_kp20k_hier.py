#!/usr/bin/anaconda3/bin/python
import os
import sys
sys.path.append(os.getcwd())

import keras
print(keras.__version__)
import numpy as np
from math import log
from prep_moduls_wo import *
from evaluation_moduls import *
from beam_tree import Node


import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Lambda, GRU, concatenate
import keras.backend as K
from keras.models import load_model
from keras.callbacks import Callback

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


DATA = 'data'
MODELS = 'models'
RESULTS = 'results'


def binarize(x, sz=50004):
  import tensorflow as tf
  return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 50004

def randomize_keyphrases(y_in):
  y_ids = []
  for i, y in enumerate(y_in):
    n_keyphrases = len(y)
    rand_id = random.randrange(n_keyphrases)
    y_ids.append(rand_id)
  return y_ids


def roundup(x):
    return x if x % 5 == 0 else x + 5 - x % 5

if __name__ == '__main__':

	
    # training and validation set
    X_train = readPickle(os.path.join(DATA,'X_train_hier.pkl'))
    X_valid = readPickle(os.path.join(DATA,'X_valid_hier.pkl'))

    y_train_in = readPickle(os.path.join(DATA,'y_train_in_hier.pkl'))
    y_train_out = readPickle(os.path.join(DATA,'y_train_out_hier.pkl'))

    y_valid_in = readPickle(os.path.join(DATA,'y_valid_in_hier.pkl'))
    y_valid_out = readPickle(os.path.join(DATA,'y_valid_out_hier.pkl'))


    ############

    all_x_train = np.concatenate((X_train, X_valid))
    all_y_train_in = np.concatenate((y_train_in, y_valid_in))
    all_y_train_out = np.concatenate((y_train_out, y_valid_out))

    savePickle(all_x_train, os.path.join(DATA,'all_x_train_hier'))
    savePickle(all_y_train_in, os.path.join(DATA,'all_y_train_in_hier'))
    savePickle(all_y_train_out, os.path.join(DATA,'all_y_train_out_hier'))
  
    ############

    docid_pair_train = []
    x_pair_train = []
    y_pair_train = []
    y_pair_train_out = []

    for i, (y_in, y_out) in enumerate(zip(all_y_train_in, all_y_train_out)):
      for j in range(len(y_in)):
        docid_pair_train.append(i)
        x_pair_train.append(all_x_train[i])
        y_pair_train.append(y_in[j])
        y_pair_train_out.append(y_out[j])

    x_pair_train = np.array(x_pair_train)
    y_pair_train = np.array(y_pair_train)
    y_pair_train_out = np.array(y_pair_train_out)

    savePickle(docid_pair_train, os.path.join(DATA,'docid_pair_train_hier'))
    savePickle(x_pair_train, os.path.join(DATA,'x_pair_train_hier'))
    savePickle(y_pair_train, os.path.join(DATA,'y_pair_train_hier'))
    savePickle(y_pair_train_out, os.path.join(DATA,'y_pair_train_out_hier'))


