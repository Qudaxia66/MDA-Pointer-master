from __future__ import unicode_literals, print_function, division
# import sys
# sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random
# 句子encoder
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from torch.autograd import Variable
use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)
        
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)


        self.lstm_sent = nn.LSTM(config.emb_sentence, config.emb_sentence_hidden, num_layers=1, batch_first= True, bidirectional=True)
        init_lstm_wt(self.lstm_sent)
        self.sentence_model = SentenceTransformer(config.sentence_model_path)
        self.emedding_linear = nn.Linear(768,128)

        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.W_h_sent = nn.Linear(config.emb_sentence_hidden*2, config.emb_sentence_hidden * 2, bias=False)

    def forward(self, input, seq_lens,enc_sentence_batch, seq_sent_lens):

        embedded = self.embedding(input)

        list_embedding = []
        list_padding = []
        for i in range(len(enc_sentence_batch)):
            sentence = enc_sentence_batch[i]
            len_sentence = len(sentence)

            if len_sentence >= config.max_enc_steps_sentence:
                sentence_input = sentence[:config.max_enc_steps_sentence]
                sentence_mask = torch.ones(len(sentence_input)).cuda()
                sentence_embedding = self.sentence_model.encode(sentence_input)

            else:
                sentence_embedding = self.sentence_model.encode(sentence)
                sentence_mask_ = torch.ones(len(sentence))

                pad_len = config.max_enc_steps_sentence - len_sentence
                sentence_mask_pad = torch.zeros(pad_len)

                pad_embeddings = []
                for i in range(pad_len):
                    pad_embeddings.append(np.zeros(config.emb_sentence))
                pad_embeddings = np.array(pad_embeddings)

                sentence_embedding = np.concatenate((sentence_embedding, pad_embeddings), axis=0)
                sentence_mask = torch.cat((sentence_mask_,sentence_mask_pad), 0).cuda()


            sentence_embedding = torch.tensor(sentence_embedding,dtype=torch.float).cuda()

            list_embedding.append(sentence_embedding)
            list_padding.append(sentence_mask)

        embedded_sentence = torch.stack(list_embedding, dim=0)
        sentence_padding_batch = torch.stack(list_padding, dim=0)

        packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)  #

        for i in range(len(seq_sent_lens)):
            seq_sent_lens[i] = config.max_enc_steps_sentence

        packed_sent = pack_padded_sequence(embedded_sentence, seq_sent_lens, batch_first=True)

        output, hidden = self.lstm(packed)
        sentence_output, sentence_hidden = self.lstm_sent(packed_sent)

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n

        encoder_sentence_outputs, _ = pad_packed_sequence(sentence_output, batch_first= True)

        encoder_outputs = encoder_outputs.contiguous()
        encoder_sentence_outputs = encoder_sentence_outputs.contiguous()

        
        encoder_feature = encoder_outputs.view(-1, 2 * config.hidden_dim)  # B * t_k x 2*hidden_dim
        encoder_sentence_feature = encoder_sentence_outputs.view(-1, 2*config.emb_sentence_hidden)

        encoder_feature = self.W_h(encoder_feature)
        encoder_sentence_feature = self.W_h_sent(encoder_sentence_feature)

        return encoder_outputs, encoder_feature, hidden, encoder_sentence_outputs, encoder_sentence_feature,\
               sentence_hidden, sentence_padding_batch

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

        # sentence
        self.reduce_h_sentence = nn.Linear(config.emb_sentence_hidden * 2, config.emb_sentence_hidden)
        init_linear_wt(self.reduce_h_sentence)
        self.reduce_c_sentence = nn.Linear(config.emb_sentence_hidden *2 , config.emb_sentence_hidden)
        init_linear_wt(self.reduce_c_sentence)

    def forward(self, hidden, sentence_hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))


        h_sentence, c_sentence = sentence_hidden
        h_in_sentence = h_sentence.transpose(0, 1).contiguous().view(-1, config.emb_sentence_hidden * 2)
        hidden_reduced_h_sentence = F.relu(self.reduce_h_sentence(h_in_sentence))
        c_in_sentence = c_sentence.transpose(0,1).contiguous().view(-1, config.emb_sentence_hidden * 2)
        hidden_reduced_c_sentence = F.relu(self.reduce_c_sentence(c_in_sentence))
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)), (hidden_reduced_h_sentence.unsqueeze(0), hidden_reduced_c_sentence.unsqueeze(0))
        # h, c dim = 1 x b x hidden_dim

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.view(-1, 1)  # B * t_k x 1
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim

            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask # B x t_k
        # print(attn_dist_)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        c_t = torch.bmm(attn_dist, encoder_outputs)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage = coverage.view(-1, t_k)
            coverage = coverage + attn_dist

        return c_t, attn_dist, coverage

class Attention_sentence(nn.Module):
    def __init__(self):
        super(Attention_sentence, self).__init__()
        # attention
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.emb_sentence_hidden * 2, bias=False)
        self.decode_proj = nn.Linear(config.emb_sentence_hidden * 2, config.emb_sentence_hidden * 2)
        self.v = nn.Linear(config.emb_sentence_hidden * 2, 1, bias=False)

    def forward(self, s_t_hat_sentence, encoder_sentence_outputs, encoder_sentence_feature,
                                                           sentence_padding_batch, coverage_sentence):
        b, t_k, n = list(encoder_sentence_outputs.size())

        dec_fea = self.decode_proj(s_t_hat_sentence)  # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()  # B x t_k x 2*hidden_dim
        dec_fea_expanded = dec_fea_expanded.view(-1, n)  # B * t_k x 2*hidden_dim


        att_features_sentence = encoder_sentence_feature + dec_fea_expanded  # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_sentence_input = coverage_sentence.view(-1, 1)  # B * t_k x 1
            coverage_sentence_feature = self.W_c(coverage_sentence_input)  # B * t_k x 2*hidden_dim

            att_features_sentence = att_features_sentence + coverage_sentence_feature

        e = F.tanh(att_features_sentence)  # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k

        attn_dist_sentence_ = F.softmax(scores, dim=1)*sentence_padding_batch   # B x t_k
        normalization_factor = attn_dist_sentence_.sum(1, keepdim=True)
        attn_dist_sentence = attn_dist_sentence_ / normalization_factor

        attn_dist_sentence = attn_dist_sentence.unsqueeze(1)  # B x 1 x t_k
        c_t_sentence = torch.bmm(attn_dist_sentence, encoder_sentence_outputs)  # B x 1 x n
        c_t_sentence = c_t_sentence.view(-1, config.emb_sentence_hidden * 2)  # B x 2*hidden_dim

        attn_dist_sentence = attn_dist_sentence.view(-1, t_k)  # B x t_k

        if config.is_coverage:
            coverage_sentence = coverage_sentence.view(-1, t_k)
            coverage_sentence = coverage_sentence + attn_dist_sentence

        return c_t_sentence, attn_dist_sentence, coverage_sentence


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_network = Attention()
        self.attention_network_sentence = Attention_sentence()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.x_context_sentence = nn.Linear(config.emb_sentence_hidden *2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)
            self.p_gen_linear_sentence = nn.Linear(config.emb_sentence_hidden * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out1_sentence = nn.Linear(config.emb_sentence_hidden * 3, config.emb_sentence_hidden)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        self.out2_sentence = nn.Linear(config.emb_sentence_hidden, config.vocab_size)
        init_linear_wt(self.out2)
        init_linear_wt(self.out2_sentence)

    def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step,
                s_t_1_sentence, encoder_sentence_outputs, encoder_sentence_feature, c_t_1_sentence, coverage_sentence,
                sentence_padding_batch):

        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_1
            h_decoder_sentence, c_decoder_sentence = s_t_1_sentence

            s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                                 c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
            s_t_hat_sentence = torch.cat((h_decoder_sentence.view(-1, config.emb_sentence_hidden),
                                          c_decoder_sentence.view(-1, config.emb_sentence_hidden)), 1)

            c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                              enc_padding_mask, coverage)
            c_t_sentence, _, coverage_next_sentence = self.attention_network_sentence\
                (s_t_hat_sentence, encoder_sentence_outputs, encoder_sentence_feature,
                                                           sentence_padding_batch, coverage_sentence)
            coverage = coverage_next
            coverage_sentence = coverage_next_sentence

        y_t_1_embd = self.embedding(y_t_1)

        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
        x1 = self.x_context_sentence(torch.cat((c_t_1_sentence, y_t_1_embd), 1))


        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        lstm_out_sentence, s_t_sentence = self.lstm(x1.unsqueeze(1), s_t_1_sentence)

        h_decoder, c_decoder = s_t
        h_decoder_sentence, c_decoder_sentence = s_t_sentence
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim
        s_t_hat_sentence = torch.cat((h_decoder_sentence.view(-1, config.emb_sentence_hidden),
                             c_decoder_sentence.view(-1, config.emb_sentence_hidden)), 1)  # B x 2*hidden_dim
        c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
                                                          enc_padding_mask, coverage)
        c_t_sentence, attn_dist_sentence, coverage_next_sentence = self.attention_network_sentence\
            (s_t_hat_sentence, encoder_sentence_outputs, encoder_sentence_feature, sentence_padding_batch, coverage_sentence)

        if self.training or step > 0:
            coverage = coverage_next
            coverage_sentence = coverage_next_sentence

        p_gen = None
        p_gen_sentence = None
        if config.pointer_gen:

            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen_input_sentence = torch.cat((c_t_sentence, s_t_hat_sentence, x1), 1)

            p_gen = self.p_gen_linear(p_gen_input)
            p_gen_sentence = self.p_gen_linear_sentence(p_gen_input_sentence)

            p_gen = F.sigmoid(p_gen)
            p_gen_sentence = F.sigmoid(p_gen_sentence)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output_sentence = torch.cat((lstm_out_sentence.view(-1, config.emb_sentence_hidden), c_t_sentence), 1)

        output = self.out1(output) # B x hidden_dim
        output_sentence = self.out1_sentence(output_sentence)

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        output_sentence = self.out2_sentence(output_sentence)
        vocab_dist = F.softmax(output, dim=1)

        vocab_dist_sentence = F.softmax(output_sentence, dim=1)



        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist * (1-p_gen_sentence)

            vocab_dist_sentence_ = p_gen_sentence * vocab_dist_sentence * p_gen

            final_vocab_dist_ = torch.add(vocab_dist_, vocab_dist_sentence_)

            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                final_vocab_dist_ = torch.cat([final_vocab_dist_, extra_zeros], 1)

            final_dist = final_vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)

        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage, s_t_sentence, c_t_sentence, attn_dist_sentence, p_gen_sentence, coverage_sentence

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        decoder = Decoder()
        reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()

        self.encoder = encoder
        self.decoder = decoder
        self.reduce_state = reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
