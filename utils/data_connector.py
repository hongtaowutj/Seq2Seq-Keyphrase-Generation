# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
import _pickle as cPickle
import h5py


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
        print(" file saved to: %s"%(os.path.join(self.filepath, self.filename)))
        f.close()

    def read_numpys(self):

        self.read_file = np.load(os.path.join(self.filepath, self.filename))


    def save_numpys(self):

        np.save(os.path.join(self.filepath, self.filename), self.data)
        print(" file saved to: %s"%(os.path.join(self.filepath, self.filename)))


    # saving file into hdf5 format
    # only works for array with same length
    
    def save_H5File(self):

        # h5Filename should be in format 'name-of-file.h5'
        h5Filename = self.filename

        # datasetName in String "" format
        datasetName = "data"

        with h5py.File(h5Filename, 'w') as hf:
            hf.create_dataset(datasetName,  data=self.data)


    # reading file in hdf5 format
    # only works for array with same length
    
    def read_H5File(self):

        # h5Filename should be in format 'name-of-file.h5'
        h5Filename = self.filename
        # datasetName in String "" format
        datasetName = "data"

        with h5py.File(h5Filename, 'r') as hf:
            data = hf[datasetName][:]

        self.read_file = data

