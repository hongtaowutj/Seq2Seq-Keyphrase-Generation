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
from keras.layers import Input, Embedding
from keras.layers import LSTM, GRU, concatenate
from keras.layers import Dense, Lambda, Reshape, Activation, Add
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from utils.sampled_softmax import SamplingLayer
from utils.predict_softmax import PredictionLayer
from utils.data_iterator import Dataiterator
from utils.data_connector import DataConnector

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SampledSoftmax():

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

	
	def train_sampled_softmax(self, pretrained_embedding, oov_embedding):

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

		fwd_encoder = GRU(self.birnn_dim, return_state=True, name='fwd_encoder')
		bwd_encoder = GRU(self.birnn_dim, return_state=True, name='bwd_encoder', go_backwards=True)
		out_encoder_1, _eh1 = fwd_encoder(enc_embedded_sequences)
		out_encoder_2, _eh2 = bwd_encoder(enc_embedded_sequences)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir_encoder')

		encoder_model = Model(inputs=in_encoder, outputs=out_bidir_encoder)
		self.encoder_model = encoder_model

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

		sampling_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='train')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: labels[:,t,:], name='label-%s'%t)(labels)
			x_dec = Lambda(lambda x: dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(dec_embedded_sequences)
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
		self.oov_lambda = oov_lambda
		self.oov_activator = oov_activator
		self.embed_decoder = embed_decoder
		self.oov_embed_decoder = oov_embed_decoder
		self.dec_embedded_sequences = dec_embedded_sequences
		self.labels = labels
		self.fwd_decoder = fwd_decoder

		return self.train_model

	def compile_(self):

		self.train_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		print("\n--- Seq2Seq (Bidirectional-GRU) with sampled softmax: trainable model --- \n")
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

		hist1_conn = DataConnector(self.filepath, 'loss_%s'%(self.filename), self.history.history['loss'])
		hist1_conn.save_pickle()

		hist2_conn = DataConnector(self.filepath, 'val_loss_%s'%(self.filename), self.history.history['val_loss'])
		hist2_conn.save_pickle()

		

	def plot_(self):

		plt.clf()
		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training', 'validation'], loc='upper right')
		plt.savefig(os.path.join(self.filepath,'loss_%s.png'%(self.filename)))


	def eval_sampled_softmax(self, stored_weights):

		eval_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='eval')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		eval_losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(self.dec_embedded_sequences)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			if t==0:
				s = self.out_bidir_encoder
			s, _ = self.fwd_decoder(x_dec, initial_state=s)
			loss = eval_softmax([s, label_t])
			eval_losses.append(loss)
			s = [s]

		eval_model = Model(inputs=[self.in_encoder, self.in_decoder, s0, self.labels], outputs=eval_losses)
		eval_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		
		eval_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		eval_model.summary()

		self.eval_model = eval_model

		return self.eval_model

	

	def predict_sampled_softmax(self, stored_weights):

		pred_softmax = PredictionLayer(self.num_samples, self.vocab_size, mode='predict')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		probs = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.dec_embedded_sequences[:,t,:], name='dec_embedding-%s'%t)(self.dec_embedded_sequences)
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

		in_state_decoder = Input(shape=(self.rnn_dim,))
		in_label = Input(shape=(None,))
		
		oov_in_decoder = self.oov_lambda(self.in_decoder)
		oov_in_decoder = self.oov_activator(oov_in_decoder)

		in_dec_embedded =  self.embed_decoder(self.in_decoder)
		oov_in_dec_embedded = self.oov_embed_decoder(oov_in_decoder)

		dec_embedded_sequences = Add()([in_dec_embedded, oov_in_dec_embedded])

		s, _ = self.fwd_decoder(dec_embedded_sequences, initial_state=[in_state_decoder])
		softmax_prob = self.pred_softmax([s, in_label])
		decoder_states = [s]
		decoder_model = Model([self.in_decoder] + [in_state_decoder] + [in_label], [softmax_prob] + decoder_states)

		self.decoder_model = decoder_model

		return self.decoder_model

	
