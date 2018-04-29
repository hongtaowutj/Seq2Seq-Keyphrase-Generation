# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle


class DataConnector():

    # input here is individual text
    def __init__(self, filepath, filename, data=None):

        self.filepath = filepath
        self.filename = filename
        self.data = data
        self.read_file = []
        
    def read_pickle(self):

        f = open(os.path.join(self.filepath, self.filename), 'rb')
        self.read_file = cPickle.load(f)
        f.close()

        return self.read_file

    def save_pickle(self):

        f = open(os.path.join(self.filepath, self.filename), 'wb')
        cPickle.dump(self.data, f)
        f.close()