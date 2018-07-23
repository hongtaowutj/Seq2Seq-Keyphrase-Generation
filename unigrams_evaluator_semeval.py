import os
import sys
sys.path.append(os.getcwd())
import numpy as np

import nltk
nltk.data.path.append('/home/TUE/inimah/nltk_data')
sno = nltk.stem.SnowballStemmer('english')

from utils.data_connector import DataConnector
from utils.beam_tree import Node
from utils.beam_decoded import BeamDecoded
from utils.evals import Evaluate
from utils.plot_metrics import Plotting


def evaluator(params):

    title = params['title']
    data_path = params['data_path']
    preprocessed_v2 = params['preprocessed_v2']
    preprocessed_data = params['preprocessed_data']
    decode_path = params['decode_path']
    result_path = params['result_path']

    idx_words = params['idx_words']
    words_idx = params['words_idx']
    decoded = params['decoded_files']
    y_1 = params['y1']
    y_2 = params['y2']

    '''
    Reading vocabulary dictionaries

    '''
    indices_words_connector = DataConnector(preprocessed_v2, idx_words, data=None)
    indices_words_connector.read_pickle()
    indices_words = indices_words_connector.read_file

    words_indices_connector = DataConnector(preprocessed_v2, words_idx, data=None)
    words_indices_connector.read_pickle()
    words_indices = words_indices_connector.read_file

    ## merge all set into one test set for trained model

    train_outputs_conn = DataConnector(data_path, y_1, data=None)
    train_outputs_conn.read_numpys()
    train_outputs = train_outputs_conn.read_file


    test_outputs_conn = DataConnector(data_path, y_2, data=None)
    test_outputs_conn.read_numpys()
    test_outputs = test_outputs_conn.read_file

    y_test_true = np.concatenate((train_outputs, test_outputs))

    print("Ground truth of keyphrases shape: %s"%str(y_test_true.shape)) # input for encoder
    sys.stdout.flush()

    '''
    Reading generated keyphrases

    '''
    # read N-generated keyphrases
    #kp_paths = ['keyphrases-beam-decode-semeval-sts-r3-%s'%(i) for i in range(244)]

    kp_paths = ['%s-%s'%(decoded, i) for i in range(244)]
    dataconn = DataConnector(filepath=None, filename=None)

    # uncomment this, for reading all generated hypothesis in one list
    #hypotheses = dataconn.read_pickles_all(result_path, kp_paths)

    # uncomment this, for reading all generated hypothesis in list of arrays
    hypotheses = dataconn.read_pickles_doc(decode_path, kp_paths)

    # on average
    n_rank = [1, 5, 10, 15, 20]

    all_rank_prediction = []
    for n in range(len(n_rank)):
        beam_predicted_keyphrases = []
        for keyphrase_list in hypotheses:

            stemmed_kps = []
            for keyphrases in keyphrase_list:

                beam_decoded = BeamDecoded(keyphrases, words_indices, indices_words, result_path)
                keyphrase = beam_decoded.decript_hypotheses()
                tokenized_keyphrase = keyphrase.split()
                #stemmed_kps = set()
                for kp in tokenized_keyphrase:
                    stemmed = sno.stem(kp)
                    if stemmed not in stemmed_kps:
                        stemmed_kps.append(stemmed)

            # print("len(decoded_kps): %s"%len(decoded_kps))
            decoded_kps = stemmed_kps[:n_rank[n]]
            beam_predicted_keyphrases.append(decoded_kps)
            #beam_predicted_keyphrases.extend(decoded_kps)
        all_rank_prediction.append(beam_predicted_keyphrases)

    '''
    Evaluating generated keyphrases of sampled softmax model + beam search decoding approach
    '''

    all_rank_acc = []
    all_rank_precision = []
    all_rank_recall = []
    all_rank_fscore = []

    all_rank_tps = []
    all_rank_fns = []
    all_rank_fps = []

    
    print("******************")
    print("Model: %s" %(title))
    print("******************")

    for i, beam_predicted in enumerate(all_rank_prediction):
        evaluate_beam = Evaluate(beam_predicted, y_test_true, result_path)

        evaluate_beam.get_true_label_list()
        
        y_true = evaluate_beam.y_true

        y_pred = evaluate_beam.y_pred


        '''

        print("length of y_true: %s"%(len(y_true)))
        print("y_true[0]: %s"%str(y_true[0]))
        print("y_true[1]: %s"%str(y_true[1]))
        print("y_true[2]: %s"%str(y_true[2]))
        print("y_true[3]: %s"%str(y_true[3]))
        print("y_true[4]: %s"%str(y_true[4]))
        print("y_true[5]: %s"%str(y_true[5]))
        print("y_true[6]: %s"%str(y_true[6]))
        print("y_true[7]: %s"%str(y_true[7]))

        '''

        

        evaluate_beam.compute_true_positive()
        evaluate_beam.compute_false_negative()
        evaluate_beam.compute_false_positive()

        evaluate_beam.compute_accuracy()
        evaluate_beam.compute_precision()
        evaluate_beam.compute_recall()
        evaluate_beam.compute_fscore()

        mean_acc, mean_precision, mean_recall, mean_fscore = evaluate_beam.compute_mean_evals()
        all_rank_acc.append(mean_acc)
        all_rank_precision.append(mean_precision)
        all_rank_recall.append(mean_recall)
        all_rank_fscore.append(mean_fscore)

        mean_tps, mean_fns, mean_fps = evaluate_beam.compute_mean_cm()
        all_rank_tps.append(mean_tps)
        all_rank_fns.append(mean_fns)
        all_rank_fps.append(mean_fps)

        print("===================")
        print("N-Rank: %s" % (n_rank[i]))

        evaluate_beam.print_mean_evals()
        evaluate_beam.print_mean_cm()

        '''

        
        tps, tp_list = evaluate_beam.compute_true_positive_all()
        fns, fn_list = evaluate_beam.compute_false_negative_all()
        fps, fp_list = evaluate_beam.compute_false_positive_all()

        all_rank_tps.append(tps)
        all_rank_fns.append(fns)
        all_rank_fps.append(fps)

        print("===================")
        print("N-Rank: %s"%(n_rank[i]))

        acc = evaluate_beam.compute_accuracy_all()
        precision = evaluate_beam.compute_precision_all()
        recall = evaluate_beam.compute_recall_all()
        fscore = evaluate_beam.compute_fscore_all()

        all_rank_acc.append(acc)
        all_rank_precision.append(precision)
        all_rank_recall.append(recall)
        all_rank_fscore.append(fscore)
        
        '''

   
    all_rank_acc_conn = DataConnector(result_path, 'all_rank_acc_unigrams', all_rank_acc)
    all_rank_acc_conn.save_pickle()

    all_rank_precision_conn = DataConnector(result_path, 'all_rank_precision', all_rank_precision)
    all_rank_precision_conn.save_pickle()

    all_rank_recall_conn = DataConnector(result_path, 'all_rank_recall_unigrams', all_rank_recall)
    all_rank_recall_conn.save_pickle()

    all_rank_fscore_conn = DataConnector(result_path, 'all_rank_fscore_unigrams', all_rank_fscore)
    all_rank_fscore_conn.save_pickle()

    all_rank_tps_conn = DataConnector(result_path, 'all_rank_tps_unigrams', all_rank_tps)
    all_rank_tps_conn.save_pickle()

    all_rank_fns_conn = DataConnector(result_path, 'all_rank_fns_unigrams', all_rank_fns)
    all_rank_fns_conn.save_pickle()

    all_rank_fps_conn = DataConnector(result_path, 'all_rank_fps_unigrams', all_rank_fps)
    all_rank_fps_conn.save_pickle()

    

    plot_metrics = Plotting(all_rank_acc, all_rank_precision, all_rank_recall, all_rank_fscore, all_rank_tps,
                            all_rank_fps, all_rank_fns, result_path)
    plot_metrics.plot_acc_fscore('plot_acc_fscore.png')
    plot_metrics.plot_metrics('plot_metrics.png')
    plot_metrics.plot_confusion_matrix('plot_confusion_matrix.png')
    
    '''
    

    all_rank_acc_conn = DataConnector(result_path, 'all_rank_acc_corpus', all_rank_acc)
    all_rank_acc_conn.save_pickle()

    all_rank_precision_conn = DataConnector(result_path, 'all_rank_precision_corpus', all_rank_precision)
    all_rank_precision_conn.save_pickle()

    all_rank_recall_conn = DataConnector(result_path, 'all_rank_recall_corpus', all_rank_recall)
    all_rank_recall_conn.save_pickle()

    all_rank_fscore_conn = DataConnector(result_path, 'all_rank_fscore_corpus', all_rank_fscore)
    all_rank_fscore_conn.save_pickle()

    all_rank_tps_conn = DataConnector(result_path, 'all_rank_tps_corpus', all_rank_tps)
    all_rank_tps_conn.save_pickle()

    all_rank_fns_conn = DataConnector(result_path, 'all_rank_fns_corpus', all_rank_fns)
    all_rank_fns_conn.save_pickle()

    all_rank_fps_conn = DataConnector(result_path, 'all_rank_fps_corpus', all_rank_fps)
    all_rank_fps_conn.save_pickle()


    plot_metrics = Plotting(all_rank_acc, all_rank_precision, all_rank_recall, all_rank_fscore, all_rank_tps,
                            all_rank_fps, all_rank_fns, result_path)
    plot_metrics.plot_acc_fscore()
    plot_metrics.plot_metrics()
    plot_metrics.plot_confusion_matrix()

    '''