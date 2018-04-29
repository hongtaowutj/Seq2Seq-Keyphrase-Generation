# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
import numpy as np
class Dataiterator():


	'''
	  1) Iteration over minibatches using next(); call reset() between epochs to randomly shuffle the data
	  2) Access to the entire dataset using all()
	'''
	
	def __init__(self, X, y_in, y_out, decoder_dim=256, batch_size=32):
		
		self.X = X # input sequences for encoder: dimension shape ( #examples, encoder_length )
		self.y_in = y_in # input sequences for decoder: dimension shape ( #examples, decoder_length )
		self.states = np.zeros((len(X), decoder_dim)) # initial state for decoder: numpy zeros with dimension shape: ( #examples, decoder_dim )
		y_labels = y_out.reshape((y_out.shape[0], y_out.shape[1], 1)) # labels for sampled softmax layer, original shape ( #examples, decoder_length ), 
		self.labels = y_labels # labels for sampled softmax layer after reshape: ( #examples, decoder_length, 1 )
		self.y_output = y_labels # model output (y_true): ( #examples, decoder_length )
		self.num_data = len(X) # total number of examples
		self.batch_size = batch_size # batch size
		self.reset() # initial: shuffling examples and set index to 0
	
	def __iter__(self): # iterates data
		
		return self


	def reset(self): # initials
		
		self.idx = 0
		self.order = np.random.permutation(self.num_data) # shuffling examples by providing randomized ids 
		
	def __next__(self): # return model inputs - outputs per batch
		
		X_ids = [] # hold ids per batch 

		while len(X_ids) < self.batch_size:

			X_id = self.order[self.idx] # copy random id from initial shuffling
			X_ids.append(X_id)

			self.idx += 1 # 
			if self.idx >= self.num_data: # exception if all examples of data have been seen (iterated)
				self.reset()
				raise StopIteration()
	
		batch_X = self.X[np.array(X_ids)] # X values (encoder input) per batch
		batch_y_in = self.y_in[np.array(X_ids)] # y_in values (decoder input) per batch
		batch_states = self.states[np.array(X_ids)] # state values (decoder state input) per batch
		batch_labels = self.labels[np.array(X_ids)] # labels (sampled softmax layer input) per batch
		batch_y_output = self.y_output[np.array(X_ids)] # y_true values (model output) per batch
		
		return batch_X, batch_y_in, batch_states, batch_labels, list(batch_y_output.swapaxes(0,1))

		  
	def all(self): # return all data examples
		
		return self.X, self.y_in, self.states, self.labels, list(self.y_output.swapaxes(0,1))

