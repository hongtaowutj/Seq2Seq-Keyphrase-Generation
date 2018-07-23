# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
import os
import sys
sys.path.append(os.getcwd())
sys.path.insert(0,'..')
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Dropout
from keras.layers import LSTM, GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import Dense, Lambda, Reshape, TimeDistributed, Activation, RepeatVector,concatenate, Concatenate, Dot
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class HierarchyAttFullSoftmax():

	def __init__(self, encoder_length, decoder_length, max_sents, embedding_dim, birnn_dim, rnn_dim, vocab_size, filepath, filename, batch_train_iter, batch_val_iter, batch_size, steps_epoch, val_steps, epochs):

		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
		self.max_sents = max_sents
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
		self.in_document = None
		self.in_decoder = None
		self.encoder_model = None
		self.decoder_model = None
		self.sent_encoder = None
		self.in_dec_embedded = None
		self.embed_decoder = None
		self.fwd_decoder = None
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

	def train_hier_att_seq2seq(self):

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

		# sentence input
		in_sentence = Input(shape=(self.encoder_length,), name='sent-input', dtype='int32')

		# document input
		in_document = Input(shape=(self.max_sents, self.encoder_length), name='doc-input', dtype='int32')

		# embedding layer
		embed_encoder = Embedding(self.vocab_size, self.embedding_dim, input_length=self.encoder_length, name='embedding-encoder')
		in_enc_embedded = embed_encoder(in_sentence)

		# CNN Block to capture N-grams features
		filter_length = [5,3,2]
		nb_filter = [16, 32, 64]
		pool_length = 2

		for i in range(len(nb_filter)):
				in_enc_embedded = Conv1D(filters=nb_filter[i],
													kernel_size=filter_length[i],
													padding='valid',
													activation='relu',
													kernel_initializer='glorot_normal',
													strides=1, name='conv_%s'%str(i+1))(in_enc_embedded)

				in_enc_embedded = Dropout(0.1, name='dropout_%s'%str(i+1))(in_enc_embedded)
				in_enc_embedded = MaxPooling1D(pool_size=pool_length, name='maxpool_%s'%str(i+1))(in_enc_embedded)

		# Bidirectional GRU to capture sentence features from CNN N-grams features
		fwd_encoder = GRU(self.birnn_dim, name='fwd-sent-encoder')
		bwd_encoder = GRU(self.birnn_dim, name='bwd-sent-encoder', go_backwards=True)
		out_encoder_1 = fwd_encoder(in_enc_embedded)
		out_encoder_2 = bwd_encoder(in_enc_embedded)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir-sent-encoder')

		 #### 1. Sentence Encoder

		sent_encoder = Model(inputs=in_sentence, outputs=out_bidir_encoder)
		self.sent_encoder = sent_encoder

		#### 2. Document Encoder
		encoded = TimeDistributed(sent_encoder, name='sent-doc-encoded')(in_document)

		# Bidirectional GRU to capture document features from encoded sentence
		fwd_doc_encoder = GRU(self.birnn_dim, return_sequences=True, name='fwd-doc-encoder')
		bwd_doc_encoder = GRU(self.birnn_dim, return_sequences=True, name='bwd-doc-encoder', go_backwards=True)
		out_encoder_doc_1 = fwd_doc_encoder(encoded)
		out_encoder_doc_2 = bwd_doc_encoder(encoded)
		out_bidir_doc_encoder = concatenate([out_encoder_doc_1, out_encoder_doc_2],axis=-1)


		#encoder_model = Model(inputs=in_document, outputs=out_bidir_doc_encoder)
		#self.encoder_model = encoder_model


		### Decoder model

		# input placeholder for teacher forcing (link ground truth to decoder input)
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


			context = one_step_attention(out_bidir_doc_encoder, s[0])
			context = Reshape((1, self.rnn_dim))(context)
			context_concat = concatenate([x_dec, context],axis=-1)
			
			'''
			end of one-step attention
			'''

			#if t==0:
			#	s = out_bidir_doc_encoder

			s, _ = fwd_decoder(context_concat, initial_state=s)
			decoder_softmax_outputs = dense_softmax(s)
			prob_outputs.append(decoder_softmax_outputs)
			s = [s]

		model = Model(inputs=[in_document, in_decoder, s0], outputs=prob_outputs)
		
		self.train_model = model
		self.in_document = in_document
		self.out_bidir_doc_encoder = out_bidir_doc_encoder
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
		print("\n--- Hierarchical Attentive Seq2Seq (CNN + Bidirectional-GRU) with full softmax: trainable model --- \n")
		self.train_model.summary()


	def train_(self):

		

		earlystop_callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(filepath=os.path.join(self.filepath,'%s.{epoch:02d}-{val_loss:.2f}.check'%(self.filename)), monitor='val_loss', save_best_only=True, save_weights_only=True),  
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


	def predict_hier_att(self, stored_weights):

		
		prediction_model = self.train_model
		prediction_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		prediction_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		prediction_model.summary()

		encoder_model = Model(inputs=self.in_document, outputs=self.out_bidir_doc_encoder)
		self.encoder_model = encoder_model

		self.prediction_model = prediction_model
		# store attention layers
		self.repeat_vector_att = prediction_model.get_layer("repeator_att")
		self.concatenator_att = prediction_model.get_layer("concator_att")
		self.densor1 = prediction_model.get_layer("densor1_att")
		self.densor2 = prediction_model.get_layer("densor2_att")
		self.att_weights = prediction_model.get_layer("attention_weights")
		self.dotor = prediction_model.get_layer("dotor_att")

		return self.prediction_model

	def create_decoder_model(self):


		# tensor/placeholder for decoder state (1 time step RNN cell)
		in_state_decoder = Input(shape=(self.rnn_dim,))

		
		# tensor/placeholder for decoder embedding output
		in_dec_embed =  self.embed_decoder(self.in_decoder)

		doc_enc_out = Input(shape=(self.max_sents, self.rnn_dim))

		'''
		attention layers
		'''
		repeat_sprev = self.repeat_vector_att(in_state_decoder)
		concat_enc_dec = self.concatenator_att([doc_enc_out, repeat_sprev])
		e_densor1 = self.densor1(concat_enc_dec)
		e_densor2 = self.densor2(e_densor1)
		alphas = self.att_weights(e_densor2)
		att_context = self.dotor([alphas, doc_enc_out])
		context = Reshape((1, self.rnn_dim))(att_context)
		x_dec_embed = Reshape((1, self.embedding_dim))(in_dec_embed)
		context_concat = concatenate([x_dec_embed, context],axis=-1)

		'''
		end of attention layers
		'''

		s, _ = self.fwd_decoder(context_concat, initial_state=[in_state_decoder])
		decoder_states = [s]
		decoder_outputs = self.dense_softmax(s)

		decoder_model = Model([self.in_decoder] + [doc_enc_out] + [in_state_decoder] , [decoder_outputs] + [alphas] + decoder_states)

		self.decoder_model = decoder_model

		return self.decoder_model

	def plot_attention_map(self, indices_words, words_indices, X, Y):
		"""
		Plot the attention map.
		"""
		# input sequences for this function is in integer format, for visualization, we need to reverse it back into its textual / string format
		text = [indices_words[i] for i in X[0]] 
		ignored_words = [words_indices['<pad>'], words_indices['<start>'],words_indices['<end>']]
		
		attention_map = np.zeros((self.decoder_length+1, self.encoder_length))
		Ty, Tx = attention_map.shape
		
		#print(Ty, Tx)
		
		x0 = Input(shape=(1, Tx))
		
		s0 = np.zeros((1, self.rnn_dim))
		att_layer = self.prediction_model.get_layer("attention_weights")
		
		x_enc_in = X
		y_dec_in = Y

		attention_model = Model(inputs=self.prediction_model.input, outputs=[att_layer.get_output_at(t) for t in range(self.decoder_length+1)]) 
		attention_weights = attention_model.predict([x_enc_in, y_dec_in, s0])
		
		for t in range(Ty):
			for t_prime in range(Tx):
				attention_map[t][t_prime] = attention_weights[t][0,t_prime,0]

		# Normalize attention map
		# row_max = attention_map.max(axis=1)
		# attention_map = attention_map / row_max[:, None]

		prediction = self.prediction_model.predict([x_enc_in, y_dec_in, s0])
		
		predicted_text = []
		for i in range(len(prediction)):
			idx_predicted = int(np.argmax(prediction[i], axis=1))
			if idx_predicted not in ignored_words:
			  predicted_text.append(idx_predicted)
			
		#print(len(predicted_text))
		
		predicted_text = list(predicted_text)
		predicted_text = [indices_words[i] for i in predicted_text if i not in ignored_words]
		
		
		# get the lengths of the string
		input_length = len(text[:50])
		output_length = len(predicted_text) #Ty
		
		# Plot the attention_map
		plt.clf()
		f = plt.figure(figsize=(20, 10))
		ax = f.add_subplot(1, 1, 1)

		# add image
		i = ax.imshow(attention_map[:output_length,150:200], interpolation='nearest', cmap='Set3')

		# add colorbar
		cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
		cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
		cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

		# add labels
		ax.set_yticks(range(output_length))
		ax.set_yticklabels(predicted_text[:output_length])

		ax.set_xticks(range(input_length))
		ax.set_xticklabels(text[(3*input_length):(4*input_length)], rotation=90)
		#ax.set_xticklabels([i for i in range(input_length)], rotation=90)
		

		ax.set_xlabel('Input Sequence')
		ax.set_ylabel('Output Sequence')

		# add grid and legend
		ax.grid()

		#f.show()
		
		return attention_map





	   