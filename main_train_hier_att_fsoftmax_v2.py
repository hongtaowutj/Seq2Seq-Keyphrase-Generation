#!/usr/bin/anaconda3/bin/python

# -*- coding: utf-8 -*-
# author: @inimah
# date: 25.04.2018

import os
import sys
sys.path.append(os.getcwd())

config = {
    'data_path': 'data/kp20k',
    'model_path':'models',
    'result_path': 'results/kp20k/v2/hier-att-full',
    'decode_path': 'results/kp20k/v2/hier-att-full/decoding',
    'preprocessed_data': 'results/kp20k/v2/data',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'glove_path': 'results/kp20k/v2/sts',
    'glove_embedding': 'nontrainable_embeddings_sent_fsoftmax.pkl',
    'oov_embedding': 'trainable_embeddings_sent_fsoftmax.pkl',

    'decoded_files': 'keyphrases-beam-sts-kp20k-hier-att-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
    'y_true': 'test_sent_output_tokens.npy',

    'birnn_dim': 150,
    'rnn_dim': 300,
    'embedding_dim': 100,
    'encoder_length': 20,
    'decoder_length' : 8,
    'max_sents' : 20,
    'batch_size': 128,
    'epoch': 100,
    'vocab_size': 10004,
    'file_name' : 'sts-kp20k-hier-att-fsoftmax-v2',
    'weights' : 'sts-kp20k-hier-att-fsoftmax-v2.03-12.23.check'
  
}

if __name__ == '__main__':

    '''
    import trainer_hier_att_fsoftmax_v2
    trainer_hier_att_fsoftmax_v2.trainer(config)
    
    
   

    import decoder_hier_att_fsoftmax_v2
    decoder_hier_att_fsoftmax_v2.decoder(config)

    '''

    import evaluator
    evaluator.evaluator(config)


    import read_kp_kp20k
    read_kp_kp20k.reader(config)

    