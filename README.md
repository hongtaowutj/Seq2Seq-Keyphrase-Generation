# Seq2Seq with Sampled Softmax for Keyphrase Generation

Introduction
==========

We train Sequence-to-Sequence (Seq2Seq) models on large target vocabulary corpus to generate multiple keyphrase-based topics of documents. The data being used to train the model are:

One training dataset (**KP20k**), five testing datasets (**KP20k, NUS, SemEval, Krapivin**).

About How to Reproduce
======================

This model has been trained on Python 3.6.4 , Keras 2.1.5, and Tensorflow 1.4.1

Information about directory and files:

* `main.py` : main file to call all functions
* `preprocessor.py` : call preprocessing class
* `trainer.py` : initiate, train, evaluate model and call decoding class to generate keyphrases
* `evaluator.py` : call evaluation class, to evaluate decoding / inference stage of text generation  
* `models/` : include all models
* `utils/` : include all class modules
* `utils/data_connector.py` : class for reading pickles / stored files
* `utils/reading_files.py` : class for reading raw text data 
* `utils/data_preprocessing.py` : class for preprocessing text data
* `utils/indexing.py` : class for generating vocabulary dictionary 
* `utils/data_iterator.py` : class for preparing per batch training set 
* `utils/sampled_softmax.py` : class for training and evaluation mode of sampling softmax layer 
* `utils/predict_softmax.py` : class for prediction mode of sampling softmax layer
* `utils/beam_tree.py` : class of tree node for storing beam search decoding 
* `utils/decoding.py` : class for decoding (generating keyphrases)
* `utils/beam_decoded.py` : class for retrieving generated keyphrases
* `utils/true_keyphrases.py` : class for retriving y_true and the statistics (mean, standard deviation of number of keyphrases per document) for deciding `beam-width` 
* `utils/evals.py` : class for computing evaluation metrics

