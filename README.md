## pytorch implementation and enforcement of Multi-domain attention pointer-generator network

## How to run training:
1) Follow data generation instruction from https://github.com/abisee/cnn-dailymail
2) Run train_word_sen.py, you might need to change some path and parameters in data_util/config.py
3) For training run train_word_sen.py, for decoding run decode_word_sen.py, and for evaluating run eval_word_sen.py

* You need to setup [pyrouge](https://github.com/andersjo/pyrouge) to get the rouge score


## Reference
[1]See A, Liu P J, Manning C D. Get To The Point: Summarization with Pointer-Generator Networks[C]//Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017: 1073-1083.
[2] pytorch implementation of Get To The Point: Summarization with Pointer-Generator Networks, https://github.com/atulkum/pointer_summarizer
