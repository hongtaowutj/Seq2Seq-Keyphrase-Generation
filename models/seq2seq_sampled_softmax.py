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
from keras.layers import Dense, Lambda, Reshape
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical

from utils.sampled_softmax import SamplingLayer
from utils.predict_softmax import PredictionLayer
from utils.beam_tree import Node
from utils.evals import Evaluate
from utils.data_iterator import Dataiterator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class SampledSoftmax():

	def __init__(self, encoder_length, decoder_length, embedding_dim, birnn_dim, rnn_dim, vocab_size, num_samples, filepath, batch_iter, batch_size):

		self.encoder_length = encoder_length
		self.decoder_length = decoder_length
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.birnn_dim = birnn_dim
		self.rnn_dim = rnn_dim
		self.num_samples = num_samples
		self.filepath = filepath
		self.batch_iter = batch_iter
		self.batch_size = batch_size
		# for storing trained graph models
		self.in_encoder = 0
		self.in_decoder = 0
		self.encoder_model = 0
		self.decoder_model = 0
		self.in_dec_embedded = 0
		self.embed_decoder = 0
		self.labels = 0
		self.fwd_decoder = 0
		self.pred_softmax = 0
		self.train_model = 0
		self.history = 0
		self.eval_model = 0
		self.perplexity_model = 0
		self.prediction_model = 0
		self.pred_softmax = 0

	def train_sampled_softmax(self):

		### Encoder model

		in_encoder = Input(shape=(self.encoder_length,), dtype='int32', name='encoder_input')
		embed_encoder = Embedding(self.vocab_size, self.embedding_dim, input_length=self.encoder_length, name='embedding_encoder')
		in_enc_embedded = embed_encoder(in_encoder)

		fwd_encoder = GRU(self.birnn_dim, return_state=True, name='fwd_encoder')
		bwd_encoder = GRU(self.birnn_dim, return_state=True, name='bwd_encoder', go_backwards=True)
		out_encoder_1, _eh1 = fwd_encoder(in_enc_embedded)
		out_encoder_2, _eh2 = bwd_encoder(in_enc_embedded)
		out_bidir_encoder = concatenate([out_encoder_1, out_encoder_2], axis=-1, name='bidir_encoder')

		encoder_model = Model(inputs=in_encoder, outputs=out_bidir_encoder)
		self.encoder_model = encoder_model

		### Decoder model

		in_decoder = Input(shape=(None, ), name='decoder_input', dtype='int32')
		embed_decoder = Embedding(self.vocab_size, self.embedding_dim, name='embedding_decoder')
		in_dec_embedded = embed_decoder(in_decoder)

		labels = Input((self.decoder_length+1,1), dtype='int32', name='labels_')

		fwd_decoder = GRU(self.rnn_dim, return_state=True)

		sampling_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='train')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

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
		print("\n--- Seq2Seq (Bidirectional-GRU) with sampled softmax): trainable model --- \n")
		self.train_model.summary()


	def train_(self):

		def gen(self.batch_size):

			while True:

				batches = [[[X, y_in, states, labels], y_output] for X, y_in, states, labels, y_output in self.batch_iter]

				for batch in batches:
					yield batch
	
		self.history = self.train_model.fit_generator(gen(self.batch_size), steps_per_epoch=steps_epoch, epochs = 10)

		self.train_model.save(os.path.join(self.filepath,'sts_sampled_softmax.h5'))
		self.train_model.save_weights(os.path.join(self.filepath,'weights_sts_sampled_softmax.hdf5'))

	def plot_(self):

		plt.clf()
		plt.plot(self.history.history['loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training'], loc='upper right')
		plt.savefig(os.path.join(self.filepath,'loss_sts_sampled_softmax.png'))


	def eval_sampled_softmax(self):

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
			s, _ = self.fwd_decoder(x_dec, initial_state=s)
			loss = eval_softmax([s, label_t])
			eval_losses.append(loss)
			s = [s]

		eval_model = Model(inputs=[self.in_encoder, self.in_decoder, s0, self.labels], outputs=eval_losses)
		eval_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		eval_model.load_weights(os.path.join(self.filepath, 'weights_sts_sampled_softmax.hdf5'))
		eval_model.summary()

		self.eval_model = eval_model

		return self.eval_model

	def perplexity_loss(self):

		perplexity_softmax = SamplingLayer(self.num_samples, self.vocab_size, mode='perplexity')

		s0 = Input(shape=(self.rnn_dim,), name='s0')
		s = [s0]

		perplexity_losses = []
		for t in range(self.decoder_length+1):

			label_t = Lambda(lambda x: self.labels[:,t,:], name='label-%s'%t)(self.labels)
			x_dec = Lambda(lambda x: self.in_dec_embedded[:,t,:], name='dec_embedding-%s'%t)(self.in_dec_embedded)
			x_dec = Reshape((1, self.embedding_dim))(x_dec)

			if t==0:
				s = self.out_bidir_encoder
			s, _ = self.fwd_decoder(x_dec, initial_state=s)
			loss = perplexity_softmax([s, label_t])
			perplexity_losses.append(loss)
			s = [s]

		perplexity_model = Model(inputs=[self.in_encoder, self.in_decoder, s0, self.labels], outputs=perplexity_losses)
		perplexity_model.compile(loss=lambda y_true, loss: loss, optimizer='rmsprop')
		perplexity_model.load_weights(os.path.join(self.filepath, 'weights_sts_sampled_softmax.hdf5'))
		perplexity_model.summary()

		self.perplexity_model = perplexity_model

		return self.perplexity_model

	def predict_sampled_softmax(self):

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
		prediction_model.load_weights(os.path.join(self.filepath, 'weights_sts_sampled_softmax.hdf5'))
		prediction_model.summary()

		self.prediction_model = prediction_model
		self.pred_softmax = pred_softmax

		return self.prediction_model

	def create_decoder_model(self):

		in_state_decoder = Input(shape=(self.rnn_dim,))
		in_label = Input(shape=(None,))
		in_dec_embed =  self.embed_decoder(self.in_decoder)

		s, _ = self.fwd_decoder(in_dec_embed, initial_state=[in_state_decoder])
		softmax_prob = self.pred_softmax([s, in_label])
		decoder_states = [s]
		decoder_model = Model([self.in_decoder] + [in_state_decoder] + [in_label], [softmax_prob] + decoder_states)

		self.decoder_model = decoder_model

		return self.decoder_model

	






	   