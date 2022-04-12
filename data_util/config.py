import os

root_dir = os.path.expanduser("~")

root_dir = "/home/aclab/PycharmProjects/cx/MDA-Pointer/"


# train_data_path = os.path.join(root_dir, "data/finished_files_en/chunked/train_*")
train_data_path = os.path.join(root_dir, "data/finished_files_en/chunked/train_*")
eval_data_path = os.path.join(root_dir, "data/finished_files_en/val.bin")

decode_data_path = os.path.join(root_dir, "data/finished_files_en/test.bin")
vocab_path = os.path.join(root_dir, "data/finished_files_en/vocab")

log_root = os.path.join(root_dir, "data/log_sentence_pad")
train_model_path = os.path.join(log_root, "final_sentence/two_batch_32/coverage")
test_data_path = os.path.join(log_root,'final_sentence/two_batch_32/decoder_result/coverage')
log_root_eval = os.path.join(log_root,"final_sentence/two_batch_32/eval_result/coverage")

# test_data_path = os.path.join(log_root,'final_sentence/decoder_result_coverage/lr_0.00001')


# Hyperparameters
hidden_dim = 256
emb_dim = 128
batch_size = 32
max_enc_steps = 400
max_dec_steps =100
max_dec_steps_test = 120
beam_size = 4
min_dec_steps = 35
vocab_size = 50000


lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
is_coverage = True
# is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12

max_iterations = 500000

use_gpu = True


lr_coverage = 0.15


emb_sentence = 768
emb_sentence_hidden = 256
max_enc_steps_sentence = 30
sentence_model_path = "/home/aclab/PycharmProjects/cx/pointer_sentence/sentence_transformers/paraphrase-distilroberta-base-v1"

