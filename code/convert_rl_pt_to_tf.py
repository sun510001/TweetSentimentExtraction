# -*- coding:utf-8 -*-

from tensorflow import *
from transformers import TFRobertaModel, RobertaConfig, RobertaTokenizer

if __name__ == '__main__':
    PATH = '../input/'
    dump = PATH
    config = RobertaConfig.from_pretrained("roberta-large")
    model = TFRobertaModel.from_pretrained("roberta-large")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    config.save_pretrained(dump)
    model.save_pretrained(dump)
    tokenizer.save_pretrained(dump)
