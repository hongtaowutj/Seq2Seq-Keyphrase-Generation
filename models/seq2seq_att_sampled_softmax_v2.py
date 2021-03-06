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

from utils.sampled_softmax import SamplingLayer
from utils.predict_softmax import PredictionLayer
from utils.data_iterator import Dataiterator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class AttentionSampledSoftmax():

	def __init__(self, encoder_length, decoder_length, embedding_dim, birnn_dim, rnn_dim, vocab_size, num_samples, filepath, filename, batch_train_iter, batch_val_iter, batch_size, steps_epoch, val_steps, epochs):

		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.birnn_dim = birnn_dim
		self.rnn_dim = rnn_dim
		self.num_samples = num_samples
		self.filepath = filepath
		self.filename = filename
		self.batch_train_iter = batch_train_iter
		self.batch_val_iter = batch_val_iter
		self.batch_size = batch_size
		self.steps_epoch = steps_epoch
		self.val_steps = val_steps
		self.epochs = epochs
		self.oov_size = 0
		# for storing trained graph models
		self.in_encoder = None
		self.in_decoder = None
		self.oov_lambda = None
		self.oov_activator = None
		self.encoder_model = None
		self.decoder_model = None
		self.dec_embedded_sequences = None
		self.embed_decoder = None
		self.oov_embed_decoder = None
		self.labels = None
		self.fwd_decoder = None
		self.pred_softmax = None
		self.train_model = None
		self.history = None
		self.eval_model = None
		self.perplexity_model = None
		self.prediction_model = None
		self.pred_softmax = None

		self.repeator = None
		self.repeat_vector_att = None
		self.concatenator = None
		self.concatenator_att = None
		self.densor1 = None
		self.densor2 = None
		self.att_weights = None
		self.dotor = None
		self.activator = None


	
	

	def train_att_sampled_softmax(self, pretrained_embedding, oov_embedding):

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

		self.vocab_size = pretrained_embedding.shape[0]
		self.oov_size = oov_embedding.shape[0]
		valid_words = self.vocab_size - self.oov_size

		in_encoder = Input(shape=(self.encoder_length,), dtype='int32', name='encoder_input')
		oov_in_encoder = Lambda(lambda x: x - valid_words)(in_encoder)
		oov_in_encoder = Activation('relu')(oov_in_encoder)

		embed_encoder = Embedding(self.vocab_size, self.embedding_dim, input_length=self.encoder_length, weights = [pretrained_embedding], trainable = False, name='embedding_encoder')
		oov_embed_encoder = Embedding(self.oov_size, self.embedding_dim, input_length=self.encoder_length, weights = [oov_embedding], trainable = True, name='oov_embedding_encoder')

		in_enc_embedded = embed_encoder(in_encoder)
		oov_in_enc_embedded = oov_embed_encoder(oov_in_encoder)

		# Add the embedding matrices
		enc_embedded_sequences = Add()([in_enc_embedded, oov_in_enc_embedded])

		fwd_encoder = GRU(self.birnn_dim, return_sequences=True, name='fwd_encoder')
		bwd_encoder = GRU(self.birnn_dim, return_sequences=True, name='bwd_encoder', go_backwards=True)
		out_encoder_1 = fwd_encoder(in_enc_embedded)
		out_encoder_2 = bwd_encoder(in_enc_embedded)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir_encoder')

		
		#encoder_model = Model(inputs=in_encoder, outputs=out_bidir_encoder)
		#self.encoder_model = encoder_model

		### Decoder model

		in_decoder = Input(shape=(None, ), name='decoder_input', dtype='int32')
		oov_lambda = Lambda(lambda x: x - valid_words)
		oov_activator = Activation('relu')

		oov_in_decoder = oov_lambda(in_decoder)
		oov_in_decoder = oov_activator(oov_in_decoder)

		embed_decoder = Embedding(self.vocab_size, self.embedding_dim, weights = [pretrained_embedding], trainable = False, name='embedding_decoder')
		oov_embed_decoder = Embedding(self.oov_size, self.embedding_dim, weights = [oov_embedding], trainable = True, name='oov_embedding_decoder')

		in_dec_embedded = embed_decoder(in_decoder)
		oov_in_dec_embedded = oov_embed_decoder(oov_in_decoder)

		# Add the embedding matrices
		dec_embedded_sequences = Add()([in_dec_embedded, oov_in_dec_embedded])

		labels = Input((self.decoder_length+1,1), dtype='int32', name='labels_')

		fwd_decoder = GRU(self.rnn_dim, return_state=True)

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]


		sampling_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='train')

		

		losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: labels[:,t,:], name='label-%s'%t)(labels)
			x_dec = Lambda(lambda x: dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(dec_embedded_sequences)
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
			
			loss = sampling_softmax([s, label_t])
			losses.append(loss)
			s = [s]

		model = Model(inputs=[in_encoder, in_decoder, s0, labels], outputs=losses)
		
		self.train_model = model
		self.in_encoder = in_encoder
		self.out_bidir_encoder = out_bidir_encoder
		self.in_decoder = in_decoder
		self.oov_lambda = oov_lambda
		self.oov_activator = oov_activator
		self.embed_decoder = embed_decoder
		self.oov_embed_decoder = oov_embed_decoder
		self.dec_embedded_sequences = dec_embedded_sequences
		self.labels = labels
		self.fwd_decoder = fwd_decoder

		# store attention layers
		self.repeat_vector_att = model.get_layer("repeator_att")
		self.concatenator_att = model.get_layer("concator_att")
		self.densor1 = model.get_layer("densor1_att")
		self.densor2 = model.get_layer("densor2_att")
		self.att_weights = model.get_layer("attention_weights")
		self.dotor = model.get_layer("dotor_att")


		return self.train_model

	def compile_(self):

		self.train_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		print("\n--- Attentive Seq2Seq (Bidirectional-GRU) with sampled softmax): trainable model --- \n")
		self.train_model.summary()


	def train_(self):

		# Set callback functions to early stop training and save the best model so far
		# monitor within a range of 10 epochs - if no improvement at all, stop training
		earlystop_callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(filepath=os.path.join(self.filepath,'%s.{epoch:02d}-{loss:.2f}.check'%(self.filename)), monitor='val_loss', save_best_only=True, save_weights_only=True),  
					 TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=True)]

		def train_gen(batch_size):

			while True:

				train_batches = [[[X, y_in, states, labels], y_output] for X, y_in, states, labels, y_output in self.batch_train_iter]

				for train_batch in train_batches:
					yield train_batch

		def val_gen(batch_size):

			while True:

				val_batches = [[[X, y_in, states, labels], y_output] for X, y_in, states, labels, y_output in self.batch_val_iter]

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


	def eval_att_sampled_softmax(self, stored_weights):

		def one_step_attention(a, s_prev):
			
			s_prev = self.repeat_vector_att(s_prev)
			concat = self.concatenator_att([a, s_prev])
			e = self.densor1(concat)
			energies = self.densor2(e)
			alphas = self.att_weights(energies)
			context = self.dotor([alphas, a])
		
			return context		

		eval_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='eval')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		eval_losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(self.dec_embedded_sequences)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			'''
			One step attention
			'''
			 # Perform one step of the attention mechanism to get back the context vector at step t
			context = one_step_attention(self.out_bidir_encoder, s[0])
			context = Reshape((1, self.rnn_dim))(context)
			context_concat = concatenate([x_dec, context],axis=-1)
			'''
			end of one-step attention
			'''

			#if t==0:
			#	s = self.out_bidir_encoder
			s, _ = self.fwd_decoder(context_concat, initial_state=s)
			loss = eval_softmax([s, label_t])
			eval_losses.append(loss)
			s = [s]

		eval_model = Model(inputs=[self.in_encoder, self.in_decoder, s0, self.labels], outputs=eval_losses)
		eval_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		
		eval_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		eval_model.summary()

		self.eval_model = eval_model

		return self.eval_model


	def predict_att_sampled_softmax(self, stored_weights):

		def one_step_attention(a, s_prev):
			
			s_prev = self.repeat_vector_att(s_prev)
			concat = self.concatenator_att([a, s_prev])
			e = self.densor1(concat)
			energies = self.densor2(e)
			alphas = self.att_weights(energies)
			context = self.dotor([alphas, a])
		
			return context

		pred_softmax = PredictionLayer(self.num_samples, self.vocab_size, mode='predict')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		probs = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(self.dec_embedded_sequences)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			'''
			One step attention
			'''
			 # Perform one step of the attention mechanism to get back the context vector at step t
			context = one_step_attention(self.out_bidir_encoder, s[0])
			context = Reshape((1, self.rnn_dim))(context)
			context_concat = concatenate([x_dec, context],axis=-1)
			'''
			end of one-step attention
			'''
 
			#if t==0:
			#	s = self.out_bidir_encoder

			s, _ = self.fwd_decoder(context_concat, initial_state=s)
			softmax_prob = pred_softmax([s, label_t])
			probs.append(softmax_prob)
			s = [s]

		prediction_model = Model(inputs=[self.in_encoder, self.in_decoder, s0, self.labels], outputs=probs)
		prediction_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		prediction_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		prediction_model.summary()

		encoder_model = Model(inputs=self.in_encoder, outputs=self.out_bidir_encoder)
		self.encoder_model = encoder_model

		self.prediction_model = prediction_model
		self.pred_softmax = pred_softmax
		# store attention layers
		self.repeat_vector_att = self.prediction_model.get_layer("repeator_att")
		self.concatenator_att = self.prediction_model.get_layer("concator_att")
		self.densor1 = self.prediction_model.get_layer("densor1_att")
		self.densor2 = self.prediction_model.get_layer("densor2_att")
		self.att_weights = self.prediction_model.get_layer("attention_weights")
		self.dotor = self.prediction_model.get_layer("dotor_att")

		return self.prediction_model

	def create_decoder_model(self):

		

		# tensor/placeholder for decoder state (1 time step RNN cell)
		in_state_decoder = Input(shape=(self.rnn_dim,))

		# tensor/placeholder for label of sampling softmax layer
		# it is not being used in prediction mode; 
		# but since the stored weight is trained on sampled softmax mode;
		# then the prediction model and decoder model still need to have similat structure
		in_label = Input(shape=(None,))

		oov_in_decoder = self.oov_lambda(self.in_decoder)
		oov_in_decoder = self.oov_activator(oov_in_decoder)

		in_dec_embedded =  self.embed_decoder(self.in_decoder)
		oov_in_dec_embedded = self.oov_embed_decoder(oov_in_decoder)

		dec_embedded_sequences = Add()([in_dec_embedded, oov_in_dec_embedded])

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
		x_dec_embed = Reshape((1, self.embedding_dim))(dec_embedded_sequences)
		context_concat = concatenate([x_dec_embed, context],axis=-1)

		'''
		end of attention layers
		'''
		s = in_state_decoder

		s, _ = self.fwd_decoder(context_concat, initial_state=[s])
		softmax_prob = self.pred_softmax([s, in_label])
		decoder_states = [s]
		decoder_model = Model([self.in_decoder] + [enc_out] + [in_state_decoder] + [in_label], [softmax_prob] + [alphas] + decoder_states)

		self.decoder_model = decoder_model

		return self.decoder_model

	def plot_attention_map(self, indices_words, words_indices, X, Y_in, state, Y_out, figurename):
		"""
		Plot the attention map.
		"""
		# input sequences for this function is in integer format, for visualization, we need to reverse it back into its textual / string format
		text = [indices_words[i] for i in X[0]] 
		ignored_words = [words_indices['<pad>'], words_indices['<start>'],words_indices['<end>']]
		
		attention_map = np.zeros((self.decoder_length+1, self.encoder_length))
		Ty, Tx = attention_map.shape
		
		#print(Ty, Tx)
		
		#x0 = Input(shape=(1, Tx))
		
		#state = np.zeros((1, self.rnn_dim))
		state = state
		#att_layer = self.prediction_model.get_layer("attention_weights")
		att_layer = self.prediction_model.layers[12]
		
		x_enc_in = X
		y_dec_in = Y_in
		y_dec_out = Y_out
		print("x_enc_in shape: %s"%str(x_enc_in.shape))
		print("y_dec_in shape: %s"%str(y_dec_in.shape))
		print("y_dec_out shape: %s"%str(y_dec_out.shape))
		print("state shape: %s"%str(state.shape))

		attention_model = Model(inputs=self.prediction_model.inputs, outputs=[att_layer.get_output_at(t) for t in range(self.decoder_length+1)]) 
		attention_weights = attention_model.predict([x_enc_in, y_dec_in, state, y_dec_out])
		
		for t in range(Ty):
			for t_prime in range(Tx):
				attention_map[t][t_prime] = attention_weights[t][0,t_prime,0]

		# Normalize attention map
		# row_max = attention_map.max(axis=1)
		# attention_map = attention_map / row_max[:, None]

		prediction = self.prediction_model.predict([x_enc_in, y_dec_in, state, y_dec_out])
		
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
		f.savefig(os.path.join(self.filepath,'attention_map-%s.png'%(figurename)))
		
		return attention_map

	
	   