from __future__ import unicode_literals, print_function, division
import sys
sys.path.append("..")

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse

import tensorflow as tf
import torch
from model_word_sen import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher_sent import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util_sen import get_input_from_batch, get_output_from_batch
from decode_word_sen import BeamSearch



use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)
        print("train_data_path111111", config.train_data_path)
        # train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        train_dir = os.path.join(config.train_model_path, 'train_%d' % (int(time.time())))
        print("00000train_dir0000", train_dir)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        print("00000000000model_dir00000000000", self.model_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # self.summary_writer = tf.summary.FileWriter(train_dir)
        self.summary_writer = tf.summary.create_file_writer(train_dir)
        # self.summary_writer = tf.summary.reexport_tf_summary(train_dir)

        self.R = ""

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        self.R = model_save_path
        print("save_model_path0000",model_save_path)
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        total_params = sum([param[0].nelement() for param in params])
        print('The Number of params of model: %.3f million' % (total_params / 1e6))  # million
        print('The number of params of model:', total_params)
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage,\
            enc_sentence_batch, seq_sent_lens, c_t_1_sentence, coverage_sentence = \
            get_input_from_batch(batch, use_cuda)

        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_feature, encoder_hidden,\
        encoder_sentence_outputs, encoder_sentence_feature, encoder_sentence_hidden,\
            sentence_padding_batch = self.model.encoder(enc_batch, enc_lens,
                                 enc_sentence_batch, seq_sent_lens)

        s_t_1 ,s_t_1_sentence= self.model.reduce_state(encoder_hidden, encoder_sentence_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing

            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage,\
                s_t_sentence, c_t_sentence, attn_dist_sentence, p_gen_sentence, next_coverage_sentence \
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

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)

        batch_avg_loss = sum_losses/dec_lens_var

        loss = torch.mean(batch_avg_loss)


        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            batch = self.batcher.next_batch()

            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 10
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 5000 == 0:
                self.save_model(running_avg_loss, iter)
                print("save model over")
                # print("--------------------------------")
                # print("Begin to generate:")
                # beam_Search_processor = BeamSearch(self.R)
                # beam_Search_processor.decode()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default= None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    
    train_processor = Train()
    train_processor.trainIters(config.max_iterations, args.model_file_path)
