from __future__ import unicode_literals, print_function, division

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import tensorflow as tf
import torch

from data_util import config
from data_util.batcher_sent_eval import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util_sen import get_input_from_batch, get_output_from_batch
from model_word_sen import Model

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root_eval, 'eval_%s' % (model_name))
        eval_loss_path = os.path.join(eval_dir,"eval_loss.txt")
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        # self.summary_writer = tf.summary.FileWriter(eval_dir)
        self.summary_writer = tf.summary.create_file_writer(eval_dir)

        self.model = Model(model_file_path, is_eval=True)

        self.R = eval_loss_path

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage,\
            enc_sentence_batch, seq_sent_lens, c_t_1_sentence, coverage_sentence = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden,\
            encoder_sentence_outputs, encoder_sentence_feature,encoder_sentence_hidden,\
            sentence_padding_batch = self.model.encoder(enc_batch, enc_lens, enc_sentence_batch, seq_sent_lens)
        s_t_1, s_t_1_sentence = self.model.reduce_state(encoder_hidden, encoder_sentence_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage,\
                s_t_sentence, c_t_sentence, attn_dist_sentence, p_gen_sentence,next_coverage_sentence \
                = self.model.decoder(y_t_1, s_t_1,
                                     encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                     extra_zeros, enc_batch_extend_vocab, coverage, di,
                                     s_t_1_sentence, encoder_sentence_outputs, encoder_sentence_feature,
                                     c_t_1_sentence, coverage_sentence, sentence_padding_batch)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_coverage_loss_sentence = torch.sum(torch.min(attn_dist_sentence, coverage_sentence), 1)

                step_loss = step_loss + config.cov_loss_wt * (step_coverage_loss + step_coverage_loss_sentence)

                coverage = next_coverage
                coverage_sentence = next_coverage_sentence

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        # return loss.data[0]
        return loss.item()



    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 10
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                iter, print_interval, time.time() - start, running_avg_loss))

                start = time.time()
            batch = self.batcher.next_batch()


if __name__ == '__main__':
#     model_filename = sys.argv[1]

    eval_processor = Evaluate("/home/aclab/PycharmProjects/cx/pointer_sentence/data/log_sentence_pad/final_sentence/two/train_1622536320/model/model_5000_1622543700")
    eval_processor.run_eval()


