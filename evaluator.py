import os
import sys
sys.path.append(os.getcwd())

from utils.data_connector import DataConnector
from utils.beam_tree import Node
from utils.beam_decoded import BeamDecoded
from utils.evals import Evaluate
from models.seq2seq_sampled_softmax import SampledSoftmax


def evaluator(params):

	data_path = params['data_path']
	model_path = params['model_path']
	result_path = params['result_path']

	'''
	Reading vocabulary dictionaries

	'''
	indices_words_connector = DataConnector(data_path, 'indices_words.pkl', data=None)
	indices_words_connector.read_pickle()
	indices_words = indices_words_connector.read_file

	words_indices_connector = DataConnector(data_path, 'words_indices.pkl', data=None)
	words_indices_connector.read_pickle()
	words_indices = words_indices_connector.read_file

	# y_true (true keyphrases) from test set

	y_test_true_connector = DataConnector(data_path, 'test_output_tokens.pkl', data=None)
	y_test_true_connector.read_pickle()
	y_test_true = y_test_true_connector.read_file

	'''
	Reading generated keyphrases

	'''

	greedy_decode_connector = DataConnector(data_path, 'all_greedy_keyphrases.pkl', data=None)
	greedy_decode_connector.read_pickle()
	all_greedy_keyphrases = greedy_decode_connector.read_file

	beam_decode_connector = DataConnector(data_path, 'all_beam_keyphrases.pkl', data=None)
	beam_decode_connector.read_pickle()
	all_beam_keyphrases = beam_decode_connector.read_file

	'''
	TO DO:
	Evaluating generated keyphrases of sampled softmax model + greedy (one best) search decoding approach
	Precision: From N-test samples, how many single predictions are correct?
	Evaluating the general performance of model in 1-top generation case
	'''


	'''
	Retrieve predicted keyphrases from beam tree (stored as nodes of tree)

	'''
	beam_predicted_keyphrases = []
	for keyphrase_list in all_beam_keyphrases:

		beam_decoded = BeamDecoded(keyphrase_list, words_indices, indices_words, result_path)
		beam_decoded.get_hypotheses()
		keyphrases = beam_decoded.keyphrases
		beam_predicted_keyphrases.append(keyphrases)

	'''
	Evaluating generated keyphrases of sampled softmax model + beam search decoding approach
	'''

	evaluate_beam = Evaluate(beam_predicted_keyphrases, y_test_true, result_path)

	evaluate_beam.get_true_keyphrases()

	evaluate_beam.compute_true_positive()
	evaluate_beam.compute_false_negative()
	evaluate_beam.compute_false_positive()

	evaluate_beam.compute_accuracy()
	evaluate_beam.compute_precision()
	evaluate_beam.compute_recall()
	evaluate_beam.compute_fscore()

	evaluate_beam.compute_mean_evals()
	evaluate_beam.compute_mean_cm()
	evaluate_beam.print_mean_evals()
