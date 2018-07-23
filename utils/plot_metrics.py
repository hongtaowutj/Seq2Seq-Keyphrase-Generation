from __future__ import print_function
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Plotting():

	def __init__(self, accuracy, precision, recall, fscore, tps, fps, fns, filepath):

		self.accuracy = accuracy
		self.precision = precision
		self.recall = recall
		self.fscore = fscore
		self.tps = tps
		self.fps = fps
		self.fns = fns
		self.filepath = filepath


	def plot_acc_fscore(self, filename):

		plt.clf()
		plt.plot(self.accuracy)
		plt.plot(self.fscore)
		plt.xticks((0, 1, 2, 3, 4), ("1", "5", "10", "15", "20"))
		plt.xlabel('top N-rank')
		plt.ylabel('performance')
		plt.legend(['accuracy', 'f1-score'], loc='upper right')
		plt.savefig(os.path.join(self.filepath, filename))

	def plot_metrics(self, filename):

		plt.clf()
		plt.plot(self.precision)
		plt.plot(self.recall)
		plt.plot(self.fscore)
		plt.xticks((0, 1, 2, 3, 4), ("1", "5", "10", "15", "20"))
		plt.xlabel('top N-rank')
		plt.ylabel('performance')
		plt.legend(['precision', 'recall', 'f1-score'], loc='upper right')
		plt.savefig(os.path.join(self.filepath, filename))

	def plot_confusion_matrix(self, filename):

		plt.clf()
		plt.plot(self.tps)
		plt.plot(self.fps)
		plt.plot(self.fns)
		plt.xticks((0, 1, 2, 3, 4), ("1", "5", "10", "15", "20"))
		plt.xlabel('top N-rank')
		plt.ylabel('performance')
		plt.legend(['true positive', 'false positive', 'false negative'], loc='upper right')
		plt.savefig(os.path.join(self.filepath, filename))












