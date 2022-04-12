from torch.autograd import Variable
import numpy as np
import torch
from data_util import config
from nltk.tokenize import sent_tokenize



def get_input_from_batch(batch, use_cuda):
  batch_size = len(batch.enc_lens)
  enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
  enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
  enc_lens = batch.enc_lens
  extra_zeros = None
  enc_batch_extend_vocab = None


  sentence_input = batch.original_articles
  enc_sentence_batch_arrary = []
  seq_sent_lens = []
  for i in range(len(sentence_input)):
    sentence = sentence_input[i]
    sentence = str(sentence)
    sentence_encoder_input = sent_tokenize(sentence)
    sen_len = len(sentence_encoder_input)

    enc_sentence_batch_arrary.append(sentence_encoder_input)
    seq_sent_lens.append(sen_len)
  enc_sentence_batch = np.array(enc_sentence_batch_arrary)

  if config.pointer_gen:

    enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())

    # enc_batch_extend_vocab_sentence
    # max_art_oovs is the max over all the article oov list in the batch
    if batch.max_art_oovs > 0:
      extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

  c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
  c_t_1_sentence = Variable(torch.zeros((batch_size, 2* config.emb_sentence_hidden)))

  coverage = None
  coverage_sentence = None
  if config.is_coverage:
    coverage = Variable(torch.zeros(enc_batch.size()))
    coverage_sentence = Variable(torch.zeros(batch_size, config.max_enc_steps_sentence))

  if use_cuda:
    enc_batch = enc_batch.cuda()
    enc_padding_mask = enc_padding_mask.cuda()


    if enc_batch_extend_vocab is not None:
      enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
    if extra_zeros is not None:
      extra_zeros = extra_zeros.cuda()
    c_t_1 = c_t_1.cuda()
    c_t_1_sentence = c_t_1_sentence.cuda()

    if coverage is not None:
      coverage = coverage.cuda()
      coverage_sentence = coverage_sentence.cuda()

  return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, \
         enc_sentence_batch, seq_sent_lens, c_t_1_sentence, coverage_sentence

def get_output_from_batch(batch, use_cuda):
  dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
  dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
  dec_lens = batch.dec_lens
  max_dec_len = np.max(dec_lens)
  dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

  target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

  if use_cuda:
    dec_batch = dec_batch.cuda()
    dec_padding_mask = dec_padding_mask.cuda()
    dec_lens_var = dec_lens_var.cuda()
    target_batch = target_batch.cuda()


  return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

