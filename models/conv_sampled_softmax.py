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
from keras.layers import Input, Embedding, Dense, Lambda, Dropout
from keras.layers import LSTM, GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import Reshape, Activation, RepeatVector,concatenate, Concatenate, Dot
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback

from utils.sampled_softmax import SamplingLayer
from utils.predict_softmax import PredictionLayer
from utils.data_iterator import Dataiterator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class ConvSampledSoftmax():

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

		# parameters for convolutional blocks
		self.filter_length = [5,3,2]
		self.nb_filter = [16, 32, 64]
		self.pool_length = 2

		# for storing trained graph models
		self.in_encoder = None
		self.in_decoder = None
		self.in_gru_enc = None
		self.encoder_model = None
		self.decoder_model = None
		self.in_dec_embedded = None
		self.embed_decoder = None
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


	
	

	def train_conv_sampled_softmax(self):

		'''
		calculating number of neurons after convolutional block
		'''

		n_block1 = int(((self.encoder_length - self.filter_length[0]) + 1) / self.pool_length)
		n_block2 = int(((n_block1 - self.filter_length[1]) + 1) / self.pool_length)
		n_block3 = int(((n_block2 - self.filter_length[2]) + 1) / self.pool_length)

		# this is an encoder length after convolutional block
		self.in_gru_enc = n_block3

		### Encoder model

		in_encoder = Input(shape=(self.encoder_length,), dtype='int32', name='encoder_input')
		embed_encoder = Embedding(self.vocab_size, self.embedding_dim, input_length=self.encoder_length, name='embedding_encoder')
		in_enc_embedded = embed_encoder(in_encoder)

		# CNN Block to capture N-grams features
		

		for i in range(len(self.nb_filter)):
				in_enc_embedded = Conv1D(filters=self.nb_filter[i],
													kernel_size=self.filter_length[i],
													padding='valid',
													activation='relu',
													kernel_initializer='glorot_normal',
													strides=1, name='conv_%s'%str(i+1))(in_enc_embedded)

				in_enc_embedded = Dropout(0.1, name='dropout_%s'%str(i+1))(in_enc_embedded)
				in_enc_embedded = MaxPooling1D(pool_size=self.pool_length, name='maxpool_%s'%str(i+1))(in_enc_embedded)

		

		fwd_encoder = GRU(self.birnn_dim, name='fwd_encoder')
		bwd_encoder = GRU(self.birnn_dim, name='bwd_encoder', go_backwards=True)
		out_encoder_1 = fwd_encoder(in_enc_embedded)
		out_encoder_2 = bwd_encoder(in_enc_embedded)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir_encoder')

		
		#encoder_model = Model(inputs=in_encoder, outputs=out_bidir_encoder)
		#self.encoder_model = encoder_model

		### Decoder model

		in_decoder = Input(shape=(None, ), name='decoder_input', dtype='int32')
		embed_decoder = Embedding(self.vocab_size, self.embedding_dim, name='embedding_decoder')
		in_dec_embedded = embed_decoder(in_decoder)

		labels = Input((self.decoder_length+1,1), dtype='int32', name='labels_')

		fwd_decoder = GRU(self.rnn_dim, return_state=True)

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]


		sampling_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='train')

		

		losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: labels[:,t,:], name='label-%s'%t)(labels)
			x_dec = Lambda(lambda x: in_dec_embedded[:,t,:], name='dec_embedding-%s'%t)(in_dec_embedded)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)


			if t==0:
				s = out_bidir_encoder
			s, _ = fwd_decoder(x_dec, initial_state=s)
			loss = sampling_softmax([s, label_t])
			losses.append(loss)
			s = [s]

		model = Model(inputs=[in_encoder, in_decoder, s0, labels], outputs=losses)
		
		self.train_model = model
		self.in_encoder = in_encoder
		self.out_bidir_encoder = out_bidir_encoder
		self.in_decoder = in_decoder
		self.embed_decoder = embed_decoder
		self.in_dec_embedded = in_dec_embedded
		self.labels = labels
		self.fwd_decoder = fwd_decoder

		return self.train_model

	def compile_(self):

		self.train_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		print("\n--- Seq2Seq (Conv + Bidirectional-GRU) with sampled softmax): trainable model --- \n")
		self.train_model.summary()


	def train_(self):

		class TimeHistory(Callback):

			def __init__(self):
				self.times = []
				self.epoch_time_start = None

			def on_train_begin(self, logs={}):
				self.times = []

			def on_epoch_begin(self, batch, logs={}):
				self.epoch_time_start = time.time()

			def on_epoch_end(self, batch, logs={}):
				self.times.append(time.time() - self.epoch_time_start)

		time_callback = TimeHistory()

		# Set callback functions to early stop training and save the best model so far
		# monitor within a range of 10 epochs - if no improvement at all, stop training
		earlystop_callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(filepath=os.path.join(self.filepath,'%s.{epoch:02d}-{loss:.2f}.check'%(self.filename)), monitor='val_loss', save_best_only=True, save_weights_only=True),  
					 TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=True), time_callback]

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

	def plot_time(self):

		plt.clf()
		plt.plot(self.history.history['times'])
		plt.title('running time')
		plt.ylabel('time')
		plt.xlabel('epoch')
		plt.legend(['training'], loc='upper right')
		plt.savefig(os.path.join(self.filepath,'time_%s.png'%(self.filename)))

	def eval_conv_sampled_softmax(self, stored_weights):

		eval_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='eval')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		eval_losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.in_dec_embedded[:,t,:], name='dec_embedding-%s'%t)(self.in_dec_embedded)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			if t==0:
				s = self.out_bidir_encoder
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


	def predict_conv_sampled_softmax(self, stored_weights):

	
		pred_softmax = PredictionLayer(self.num_samples, self.vocab_size, mode='predict')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		probs = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.in_dec_embedded[:,t,:], name='dec_embedding-%s'%t)(self.in_dec_embedded)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			
 
			if t==0:
				s = self.out_bidir_encoder

			s, _ = self.fwd_decoder(x_dec, initial_state=s)
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
		

		return self.prediction_model

	def create_decoder_model(self):

		# tensor/placeholder for decoder input (1 word)
		in_decoder = Input(shape=(1, ), name='decoder_input', dtype='int32')

		# tensor/placeholder for decoder state (1 time step RNN cell)
		in_state_decoder = Input(shape=(self.rnn_dim,))

		# tensor/placeholder for label of sampling softmax layer
		# it is not being used in prediction mode; 
		# but since the stored weight is trained on sampled softmax mode;
		# then the prediction model and decoder model still need to have similat structure
		in_label = Input(shape=(None,))

		# tensor/placeholder for decoder embedding output
		in_dec_embed =  self.embed_decoder(in_decoder)

		s, _ = self.fwd_decoder(in_dec_embed, initial_state=[in_state_decoder])
		softmax_prob = self.pred_softmax([s, in_label])
		decoder_states = [s]
		decoder_model = Model([in_decoder] + [in_state_decoder] + [in_label], [softmax_prob] + decoder_states)

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

	
	   