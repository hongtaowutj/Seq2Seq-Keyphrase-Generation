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
from keras.layers import Input, Embedding, Dropout
from keras.layers import LSTM, GRU, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import Dense, Lambda, Reshape, TimeDistributed, concatenate
import keras.backend as K
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback


#from utils.data_iterator import Dataiterator

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


class HierarchyFullSoftmax():

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
		self.decoder_dense = 0
		self.train_model = None
		self.history = None

	def train_hier_seq2seq(self):

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
		fwd_doc_encoder = GRU(self.birnn_dim, return_state=True, name='fwd-doc-encoder')
		bwd_doc_encoder = GRU(self.birnn_dim, return_state=True, name='bwd-doc-encoder', go_backwards=True)
		out_encoder_doc_1, doc_eh_1 = fwd_doc_encoder(encoded)
		out_encoder_doc_2, doc_eh_2 = bwd_doc_encoder(encoded)
		out_bidir_doc_encoder = concatenate([doc_eh_1, doc_eh_2],axis=-1)


		encoder_model = Model(inputs=in_document, outputs=out_bidir_doc_encoder)
		self.encoder_model = encoder_model


		### Decoder model

		# input placeholder for teacher forcing (link ground truth to decoder input)
		in_decoder = Input(shape=(None, ), name='decoder_input', dtype='int32')
		embed_decoder = Embedding(self.vocab_size, self.embedding_dim, name='embedding_decoder')
		in_dec_embedded = embed_decoder(in_decoder)

		fwd_decoder = GRU(self.rnn_dim, return_sequences=True, return_state=True, name='fwd-decoder')

		dec_outputs, dec_state_h = fwd_decoder(in_dec_embedded, initial_state=out_bidir_doc_encoder)
		decoder_dense = Dense(self.vocab_size, activation='softmax', name='prediction-layer')
		decoder_outputs = decoder_dense(dec_outputs)

		model = Model(inputs=[in_document, in_decoder], outputs=decoder_outputs)
		
		self.train_model = model
		self.in_document = in_document
		self.out_bidir_doc_encoder = out_bidir_doc_encoder
		self.in_decoder = in_decoder
		self.embed_decoder = embed_decoder
		self.in_dec_embedded = in_dec_embedded
		self.fwd_decoder = fwd_decoder
		self.decoder_dense = decoder_dense

		return self.train_model

	def compile_(self):

		self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		print("\n--- Hierarchical Seq2Seq (CNN + Bidirectional-GRU) with full softmax): trainable model --- \n")
		self.train_model.summary()


	def train_(self):


		earlystop_callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(filepath=os.path.join(self.filepath,'%s.{epoch:02d}-{val_loss:.2f}.check'%(self.filename)), monitor='val_loss', save_best_only=True, save_weights_only=True),  
					 TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=True)]

		def train_gen(batch_size):

			while True:

				train_batches = [[[X, y_in], y_output] for X, y_in, y_output in self.batch_train_iter]

				for train_batch in train_batches:
					yield train_batch
	
		def val_gen(batch_size):

			while True:

				val_batches = [[[X, y_in], y_output] for X, y_in, y_output in self.batch_val_iter]

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

	def predict_seq2seq(self, stored_weights):

		prediction_model = self.train_model
		prediction_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		prediction_model.load_weights(os.path.join(self.filepath, '%s'%(stored_weights)))
		prediction_model.summary()

		encoder_model = Model(inputs=self.in_document, outputs=self.out_bidir_doc_encoder)
		self.encoder_model = encoder_model

		self.prediction_model = prediction_model

		return self.prediction_model


	def create_decoder_model(self):

		in_state_decoder = Input(shape=(self.rnn_dim,))
		in_dec_embed =  self.embed_decoder(self.in_decoder)

		outputs, state_h = self.fwd_decoder(in_dec_embed, initial_state=[in_state_decoder])
		decoder_states = [state_h]
		decoder_outputs = self.decoder_dense(outputs)
		decoder_model = Model([self.in_decoder] + [in_state_decoder] , [decoder_outputs] + decoder_states)

		decoder_model.summary()

		self.decoder_model = decoder_model

		return self.decoder_model

	






	   