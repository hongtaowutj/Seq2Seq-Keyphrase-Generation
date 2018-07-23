#!/usr/bin/anaconda3/bin/python
# -*- coding: utf-8 -*-
# author: @inimah


import os
import sys
sys.path.append(os.getcwd())


# standar seq2seq
config_sts_ss_e1 = {
    'title': 'sts_ss_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/sts',
    'decode_path': 'results/inspec/v1/sts/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-sts-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_ss_e2 = {
    'title': 'sts_ss_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/sts',
    'decode_path': 'results/inspec/v2/sts/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-sts-v2',
    'idx_words': 'all_idxword_vocabulary.pkl',
    'words_idx': 'all_wordidx_vocabulary.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_fs_e1 = {
    'title': 'sts_fs_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/sts-full',
    'decode_path': 'results/inspec/v1/sts-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-sts-fsoftmax-v1',
    'idx_words': 'all_indices_words_fsoftmax.pkl',
    'words_idx': 'all_words_indices_fsoftmax.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_fs_e2 = {
    'title': 'sts_fs_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/sts-full',
    'decode_path': 'results/inspec/v2/sts-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-sts-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_fsoftmax.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}


config_sts_fs_att_e1 = {
    'title': 'sts_fs_att_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/sts-att-full',
    'decode_path': 'results/inspec/v1/sts-att-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-att-fsoftmax-v1',
    'idx_words': 'all_indices_words_fsoftmax.pkl',
    'words_idx': 'all_words_indices_fsoftmax.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_fs_att_e2 = {
    'title': 'sts_fs_att_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/sts-att-full',
    'decode_path': 'results/inspec/v2/sts-att-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-att-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_fsoftmax.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_ss_att_e1 = {
    'title': 'sts_ss_att_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/sts-att',
    'decode_path': 'results/inspec/v1/sts-att/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-att-v1',
    'idx_words': 'all_indices_words.pkl',
    'words_idx': 'all_words_indices.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

config_sts_ss_att_e2 = {
    'title': 'sts_ss_att_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/sts-att',
    'decode_path': 'results/inspec/v2/sts-att/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-att-v2',
    'idx_words': 'all_idxword_vocabulary.pkl',
    'words_idx': 'all_wordidx_vocabulary.pkl',
    'y1': 'train_output_tokens.npy',
    'y2': 'val_output_tokens.npy',
    'y3': 'test_output_tokens.npy'
}

# hierarchical seq2seq

config_hier_ss_e1 = {
    'title': 'hier_ss_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/hier',
    'decode_path': 'results/inspec/v1/hier/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-v1',
    'idx_words': 'all_indices_words_sent.pkl',
    'words_idx': 'all_words_indices_sent.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_ss_e2 = {
    'title': 'hier_ss_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/hier',
    'decode_path': 'results/inspec/v2/hier/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-v2',
    'idx_words': 'all_idxword_vocabulary_sent.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_fs_e1 = {
    'title': 'hier_fs_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/hier-full',
    'decode_path': 'results/inspec/v1/hier-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-fsoftmax-v1',
    'idx_words': 'all_indices_words_sent_fsoftmax.pkl',
    'words_idx': 'all_words_indices_sent_fsoftmax.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_fs_e2 = {
    'title': 'hier_fs_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/hier-full',
    'decode_path': 'results/inspec/v2/hier-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}


config_hier_fs_att_e1 = {
    'title': 'hier_fs_att_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/hier-att-full',
    'decode_path': 'results/inspec/v1/hier-att-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-att-fsoftmax-v1',
    'idx_words': 'all_indices_words_sent_fsoftmax.pkl',
    'words_idx': 'all_words_indices_sent_fsoftmax.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_fs_att_e2 = {
    'title': 'hier_fs_att_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/hier-att-full',
    'decode_path': 'results/inspec/v2/hier-att-full/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-att-fsoftmax-v2',
    'idx_words': 'all_idxword_vocabulary_sent_fsoftmax.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent_fsoftmax.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_ss_att_e1 = {
    'title': 'hier_ss_att_e1 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v1/hier-att',
    'decode_path': 'results/inspec/v1/hier-att/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v1/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-att-v1',
    'idx_words': 'all_indices_words_sent.pkl',
    'words_idx': 'all_words_indices_sent.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

config_hier_ss_att_e2 = {
    'title': 'hier_ss_att_e2 - inspec',
    'data_path': 'data/inspec',
    'result_path': 'results/inspec/v2/hier-att',
    'decode_path': 'results/inspec/v2/hier-att/decoding',
    'preprocessed_v2': 'results/kp20k/v2/data',
    'preprocessed_data': 'results/inspec/v2/data',
    'decoded_files': 'keyphrases-beam-decode-inspec-hier-att-v2',
    'idx_words': 'all_idxword_vocabulary_sent.pkl',
    'words_idx': 'all_wordidx_vocabulary_sent.pkl',
    'y1': 'train_output_sent_tokens.npy',
    'y2': 'val_output_sent_tokens.npy',
    'y3': 'test_output_sent_tokens.npy'
}

if __name__ == '__main__':

    
    import unigrams_evaluator_inspec
    unigrams_evaluator_inspec.evaluator(config_sts_fs_e1)
    unigrams_evaluator_inspec.evaluator(config_sts_fs_e2)
    unigrams_evaluator_inspec.evaluator(config_sts_ss_e1)
    unigrams_evaluator_inspec.evaluator(config_sts_ss_e2)
    unigrams_evaluator_inspec.evaluator(config_sts_fs_att_e1)
    unigrams_evaluator_inspec.evaluator(config_sts_fs_att_e2)
    unigrams_evaluator_inspec.evaluator(config_sts_ss_att_e1)
    unigrams_evaluator_inspec.evaluator(config_sts_ss_att_e2)

    unigrams_evaluator_inspec.evaluator(config_hier_fs_e1)
    unigrams_evaluator_inspec.evaluator(config_hier_fs_e2)
    unigrams_evaluator_inspec.evaluator(config_hier_ss_e1)
    unigrams_evaluator_inspec.evaluator(config_hier_ss_e2)
    unigrams_evaluator_inspec.evaluator(config_hier_fs_att_e1)
    unigrams_evaluator_inspec.evaluator(config_hier_fs_att_e2)
    unigrams_evaluator_inspec.evaluator(config_hier_ss_att_e1)
    unigrams_evaluator_inspec.evaluator(config_hier_ss_att_e2)

    
