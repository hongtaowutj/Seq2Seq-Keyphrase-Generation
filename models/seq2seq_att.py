# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0,'..')
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Lambda, Activation, Add
from keras.layers import LSTM, GRU
from keras.layers import Reshape, Activation, RepeatVector,concatenate, Concatenate, Dot
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class AttentionFullSoftmax():

	def __init__(self, encoder_length, decoder_length, embedding_dim, birnn_dim, rnn_dim, vocab_size, filepath, filename, batch_train_iter, batch_val_iter, batch_size, steps_epoch, val_steps, epochs):

		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.birnn_dim = birnn_dim
		self.rnn_dim = rnn_dim
		self.filepath = filepath
		self.filename = filename
		self.batch_train_iter = batch_train_iter
		self.batch_val_iter = batch_val_iter
		self.batch_size = batch_size
		self.steps_epoch = steps_epoch
		self.val_steps = val_steps
		self.epochs = epochs
		
		# for storing trained graph models
		self.in_encoder = None
		self.in_decoder = None
		self.encoder_model = None
		self.decoder_model = None
		self.in_dec_embedded = None
		self.embed_decoder = None
		self.fwd_decoder = None
		self.decoder_dense = None
		self.train_model = None
		self.history = None
	

		self.repeator = None
		self.repeat_vector_att = None
		self.concatenator = None
		self.concatenator_att = None
		self.densor1 = None
		self.densor2 = None
		self.att_weights = None
		self.dotor = None
		self.activator = None



	def train_att_seq2seq(self):

		# custom softmax for normalization w.r.t axis = 1
		def custom_softmax(x, axis=1):
			"""Softmax activation function.
			# Arguments
				x : Tensor.
				axis: Integer, axis along which the softmax normalization is applied.
			# Returns
				Tensor, output of softmax transformation.
			# Raises
				ValueError: In case `dim(x) == 1`.
			"""
			ndim = K.ndim(x)
			if ndim == 2:
				return K.softmax(x)
			elif ndim > 2:
				e = K.exp(x - K.max(x, axis=axis, keepdims=True))
				s = K.sum(e, axis=axis, keepdims=True)
				return e / s
			else:
				raise ValueError('Cannot apply softmax to a tensor that is 1D')

		#### Global variables for one step attention layers

		# RepeatVector layer is used to copy hidden state from decoder RNN cell at previous time step (h_dec(t-1)) --> repeated into number of sequence in encoder side
		self.repeator = RepeatVector(self.encoder_length, name='repeator_att')

		# Concatenate layer is for concatenating hidden state from encoder RNN cells (h_enc(t)) and repeated / copied hidden state from decoder side (h_dec(t-1))
		self.concatenator = Concatenate(axis=-1, name='concator_att')

		# Deep feed forward networks to generate attention weight vector per time step
		# first dense layer is to create intermediate energies (non-linear projection of learnt hidden state representation)
		self.densor1 = Dense(self.decoder_length+1, activation = "tanh", name='densor1_att')

		# second dense layer is to further create energies (non-linear projection of learnt hidden state representation)
		self.densor2 = Dense(1, activation = "relu", name='densor2_att')

		self.activator = Activation(custom_softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook

		# dot product to compute attention weight from attention vector (.) and hidden state from encoder RNN cell in one step 
		self.dotor = Dot(axes = 1, name='dotor_att')


		

		def one_step_attention(a, s_prev):
			"""
			Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
			"alphas" and the hidden states "a" of the Bi-GRU encoder.
			
			Arguments:
			a -- hidden state output of the Bi-GRU encoder, numpy-array of shape (m, Tx, 2*n_a)
			s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
			
			Returns:
			context -- context vector, input of the next (post-attetion) LSTM cell
			"""
			
			# Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
			s_prev = self.repeator(s_prev)
			print("s_prev: %s"%s_prev.shape)
			sys.stdout.flush()
			
			# Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
			concat = self.concatenator([a, s_prev])

			
			# Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
			e = self.densor1(concat)

			# Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
			
			energies = self.densor2(e)

			
			# Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
			alphas = self.activator(energies)

			
			# Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
			context = self.dotor([alphas, a])
			### END CODE HERE ###
		
			return context

		### Encoder model


		in_encoder = Input(shape=(self.encoder_length,), dtype='int32', name='encoder_input')


		embed_encoder = Embedding(self.vocab_size, self.embedding_dim, input_length=self.encoder_length, name='embedding_encoder')
		

		in_enc_embedded = embed_encoder(in_encoder)
		

		fwd_encoder = GRU(self.birnn_dim, return_sequences=True, name='fwd_encoder')
		bwd_encoder = GRU(self.birnn_dim, return_sequences=True, name='bwd_encoder', go_backwards=True)
		out_encoder_1 = fwd_encoder(in_enc_embedded)
		out_encoder_2 = bwd_encoder(in_enc_embedded)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir_encoder')

		
		encoder_model = Model(inputs=in_encoder, outputs=out_bidir_encoder)
		self.encoder_model = encoder_model

		### Decoder model

		in_decoder = Input(shape=(None, ), name='decoder_input', dtype='int32')

		embed_decoder = Embedding(self.vocab_size, self.embedding_dim, name='embedding_decoder')
		
		in_dec_embedded = embed_decoder(in_decoder)
		

		fwd_decoder = GRU(self.rnn_dim, return_state=True)
		dense_softmax = Dense(self.vocab_size,  activation='softmax')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		prob_outputs = []


		for t in range(self.decoder_length+1):

			x_dec = Lambda(lambda x: in_dec_embedded[:,t,:], name='dec_embedding-%s'%t)(in_dec_embedded)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			'''
			One step attention
			'''
			 # Perform one step of the attention mechanism to get back the context vector at step t


			context = one_step_attention(out_bidir_encoder, s[0])	
			context = Reshape((1, self.rnn_dim))(context)	
			context_concat = concatenate([x_dec, context],axis=-1)
			
			'''
			end of one-step attention
			'''

			#if t==0:
			#	s = out_bidir_encoder

			s, _ = fwd_decoder(context_concat, initial_state=s)
			decoder_softmax_outputs = dense_softmax(s)
			
			prob_outputs.append(decoder_softmax_outputs)
			s = [s]

		model = Model(inputs=[in_encoder, in_decoder, s0], outputs=prob_outputs)
		
		self.train_model = model
		self.in_encoder = in_encoder
		self.out_bidir_encoder = out_bidir_encoder
		self.in_decoder = in_decoder
		self.embed_decoder = embed_decoder
		self.in_dec_embedded = in_dec_embedded
		self.fwd_decoder = fwd_decoder
		self.dense_softmax = dense_softmax

		# store attention layers
		self.repeat_vector_att = model.get_layer("repeator_att")
		self.concatenator_att = model.get_layer("concator_att")
		self.densor1 = model.get_layer("densor1_att")
		self.densor2 = model.get_layer("densor2_att")
		self.att_weights = model.get_layer("attention_weights")
		self.dotor = model.get_layer("dotor_att")


		return self.train_model

	def compile_(self):

		self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		print("\n--- Attentive Seq2Seq (Bidirectional-GRU) with full softmax: trainable model --- \n")
		self.train_model.summary()


	def train_(self):

		# Set callback functions to early stop training and save the best model so far
		# monitor within a range of 10 epochs - if no improvement at all, stop training
		earlystop_callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(filepath=os.path.join(self.filepath,'%s.{epoch:02d}-{loss:.2f}.check'%(self.filename)), monitor='val_loss', save_best_only=True, save_weights_only=True),  
					 TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=True)]

		def train_gen(batch_size):

			while True:

				train_batches = [[[X, y_in, states], y_output] for X, y_in, states, y_output in self.batch_train_iter]

				for train_batch in train_batches:
					yield train_batch

		def val_gen(batch_size):

			while True:

				val_batches = [[[X, y_in, states], y_output] for X, y_in, states, y_output in self.batch_val_iter]

				for val_batch in val_batches:
					yield val_batch
	
		self.history = self.train_model.fit_generator(train_gen(self.batch_size), validation_data=val_gen(self.batch_size), validation_steps=self.val_steps, steps_per_epoch=self.steps_epoch, epochs = self.epochs, callbacks = earlystop_callbacks)

	def plot_(self):

		plt.clf()
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training', 'validation'], loc='upper right')
		plt.savefig(os.path.join(self.filepath,'loss_%s.png'%(self.filename)))


	
	def predict_att_seq2seq(self, stored_weights):

		
		prediction_model = self.train_model
		prediction_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		prediction_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		prediction_model.summary()

		encoder_model = Model(inputs=self.in_encoder, outputs=self.out_bidir_encoder)
		self.encoder_model = encoder_model

		self.prediction_model = prediction_model
		# store attention layers
		self.repeat_vector_att = self.prediction_model.get_layer("repeator_att")
		self.concatenator_att = self.prediction_model.get_layer("concator_att")
		self.densor1 = self.prediction_model.get_layer("densor1_att")
		self.densor2 = self.prediction_model.get_layer("densor2_att")
		self.att_weights = self.prediction_model.get_layer("attention_weights")
		self.dotor = self.prediction_model.get_layer("dotor_att")

		return self.prediction_model

	def create_decoder_model(self):

		in_state_decoder = Input(shape=(self.rnn_dim,))

		in_dec_embedded =  self.embed_decoder(self.in_decoder)
		
		# tensor/placeholder for encoder output
		enc_out = Input(shape=(self.encoder_length, self.rnn_dim))

		'''
		attention layers
		'''
		# repeator of decoder RNN state output (post attention RNN decoder) 
		repeat_sprev = self.repeat_vector_att(in_state_decoder)
		# concatenator of encoder state output and decoder state output
		concat_enc_dec = self.concatenator_att([enc_out, repeat_sprev])
		# dense layer to project / reduce dimension to decoder length
		e_densor1 = self.densor1(concat_enc_dec)
		# reduce / project dimension to 1
		e_densor2 = self.densor2(e_densor1)
		# attention weight is softmax of energies from encoder-decoder
		alphas = self.att_weights(e_densor2)
		# context of encoder weighted by attention weight
		att_context = self.dotor([alphas, enc_out])
		context = Reshape((1, self.rnn_dim))(att_context)
		x_dec_embed = Reshape((1, self.embedding_dim))(in_dec_embedded)
		context_concat = concatenate([x_dec_embed, context],axis=-1)

		'''
		end of attention layers
		'''
		s = in_state_decoder

		s, _ = self.fwd_decoder(context_concat, initial_state=[s])
		decoder_states = [s]
		decoder_outputs = self.dense_softmax(s)

		
		decoder_model = Model([self.in_decoder] + [enc_out] + [in_state_decoder], [decoder_outputs] + [alphas] + decoder_states)

		self.decoder_model = decoder_model

		return self.decoder_model

	
	
	   