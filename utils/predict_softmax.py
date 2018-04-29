# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018
from keras.layers import Layer
import tensorflow as tf
class PredictionLayer(Layer):
	def __init__(self, num_sampled, num_classes, mode, **kwargs):
		self.num_sampled = num_sampled
		self.num_classes = num_classes
		self.mode = mode
		super(PredictionLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		dense_shape, classes_shape = input_shape
		self.kernel = self.add_weight(name='kernel',
									  shape=(self.num_classes, dense_shape[1]),
									  initializer='uniform',
									  trainable=True)
		self.bias = self.add_weight(name='bias',
									  shape=(self.num_classes,),
									  initializer='uniform',
									  trainable=True)  

		super(PredictionLayer, self).build(input_shape)  

	def call(self, inputs_and_labels):
		inputs, labels = inputs_and_labels
			
		if self.mode == "predict":
		  
		  logits = tf.matmul(inputs, tf.transpose(self.kernel))
		  logits = tf.nn.bias_add(logits, self.bias)
		  softmax = tf.nn.softmax(logits)
		  output = softmax
		  
		return output

	def compute_output_shape(self, input_shape):
		dense_shape, classes_shape = input_shape
		return (dense_shape[0], self.num_classes)