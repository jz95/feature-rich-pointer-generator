# -*- coding: utf-8 -*-
from run_summarization import *


def main(unused_argv):
    vocab = Vocab('/Users/j.zhou/mlp_project/data/finished_files', 500)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    FLAGS.how_to_use_pos = 'concate'
    FLAGS.how_to_use_char = 'concate'

    hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'pos_emb_dim', 'char_emb_dim',
                   'how_to_use_pos', 'how_to_use_char',
                   'batch_size', 'max_dec_steps', 'max_enc_steps', 'max_word_len',
                   'coverage', 'cov_loss_wt', 'pointer_gen']

    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val.value  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)


    # Create a batcher object that will create minibatches of data
    # batcher = Batcher('/Users/j.zhou/mlp_project/data/finished_files/val.bin', vocab, hps, single_pass=False)

    # batch = batcher.next_batch()
    model = SummarizationModel(hps, vocab)
    model.build_graph()
    # setup_training(model, batcher)

if __name__ == '__main__':
    tf.app.run()
