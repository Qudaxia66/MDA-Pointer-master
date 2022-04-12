#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys


import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time

import torch
from torch.autograd import Variable

from data_util.batcher_sent import Batcher, data
from data_util.data import Vocab
from data_util import config
from model_word_sen import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log
from train_util_sen import get_input_from_batch



use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, state_sentence, context, context_sentence, coverage, coverage_sentence):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.state_sentence = state_sentence
    self.context = context
    self.context_sentence = context_sentence
    self.coverage = coverage
    self.coverage_sentence = coverage_sentence

  def extend(self, token, log_prob, state, state_sentence, context, context_sentence, coverage, coverage_sentence):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      state_sentence = state_sentence,
                      context = context,
                      context_sentence = context_sentence,
                      coverage = coverage,
                      coverage_sentence = coverage_sentence)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        print("decoder0000model_name", model_name)
        # self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._decode_dir = os.path.join(config.test_data_path, 'decode_%s' % (model_name) )
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        self._rouge_article_dir = os.path.join(self._decode_dir, 'rouge_article_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir, self._rouge_article_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        print("decode_data_path00000000", config.decode_data_path)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()

        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]

            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 11490 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()
                break

            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)



    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, \
        enc_sentence_batch, seq_sent_lens, c_t_0_sentence, coverage_sentence_t_0 = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden,\
            encoder_sentence_outputs, encoder_sentence_feature, encoder_sentence_hidden,\
                sentence_padding_batch = self.model.encoder(enc_batch, enc_lens, enc_sentence_batch, seq_sent_lens)

        s_t_0, s_t_0_sentence = self.model.reduce_state(encoder_hidden, encoder_sentence_hidden)


        dec_h, dec_c = s_t_0
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        dec_h_sentence, dec_c_sentence = s_t_0_sentence
        dec_h_sentence = dec_h_sentence.squeeze()
        dec_c_sentence = dec_c_sentence.squeeze()


        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      state_sentence=(dec_h_sentence[0], dec_c_sentence[0]),
                      context = c_t_0[0],
                      context_sentence = c_t_0_sentence[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None),
                      coverage_sentence=(coverage_sentence_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps_test and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))

            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_state_h_sentence = []
            all_state_c_sentence = []

            all_context = []
            all_context_sentence = []

            for h in beams:
                state_h, state_c = h.state
                state_h_sentence,state_c_sentence = h.state_sentence
                all_state_h.append(state_h)
                all_state_c.append(state_c)


                all_state_h_sentence.append(state_h_sentence)
                all_state_c_sentence.append(state_c_sentence)

                all_context.append(h.context)
                all_context_sentence.append(h.context_sentence)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            s_t_1_sentence = (torch.stack(all_state_h_sentence, 0).unsqueeze(0), torch.stack(all_state_c_sentence, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)
            c_t_1_sentence = torch.stack(all_context_sentence, 0)

            coverage_t_1 = None
            coverage_t_1_sentence = None
            if config.is_coverage:
                all_coverage = []
                all_coverage_sentence = []
                for h in beams:
                    all_coverage.append(h.coverage)
                    all_coverage_sentence.append(h.coverage_sentence)
                coverage_t_1 = torch.stack(all_coverage, 0)
                coverage_t_1_sentence = torch.stack(all_coverage_sentence, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t,\
                s_t_sentence, c_t_sentence, attn_dist_sentence, p_gen_sentence, coverage_t_sentence\
                = self.model.decoder(y_t_1, s_t_1,
                                     encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                     extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps,
                                     s_t_1_sentence, encoder_sentence_outputs, encoder_sentence_feature,
                                     c_t_1_sentence, coverage_t_1_sentence, sentence_padding_batch)


            log_probs = torch.log(final_dist)

            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            dec_h_sentence, dec_c_sentence = s_t_sentence
            dec_h_sentence = dec_h_sentence.squeeze()
            dec_c_sentence = dec_c_sentence.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                state_i_sentence = (dec_h_sentence[i], dec_c_sentence[i])
                context_i = c_t[i]
                context_i_sentence = c_t_sentence[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)
                coverage_i_sentence = (coverage_t_sentence[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   state_sentence=state_i_sentence,
                                   context=context_i,
                                   context_sentence=context_i_sentence,
                                   coverage=coverage_i,
                                   coverage_sentence=coverage_i_sentence)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    # model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch("/home/aclab/PycharmProjects/cx/pointer_sentence/data/log_sentence_pad/final_sentence/two/train_1622536320/model/model_5000_1622543700")
    beam_Search_processor.decode()


